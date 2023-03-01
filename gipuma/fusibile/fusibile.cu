/* vim: ft=cpp
 * */

#include <stdio.h>
#include "globalstate.h"
#include "camera.h"
#include "config.h"

#include <vector_types.h>		// float4
#include <cuda.h>
#include <curand_kernel.h>
#include "point_cloud_list.h"
// #include <iostream>

#define FORCEINLINE __forceinline__

static __device__ float4 operator-(float4 a, float4 b)
{
	return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, 0);
}

static __device__ float4 operator+(float4 a, float4 b)
{
	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, 0);
}

static __device__ float4 operator/(float4 a, float k)
{
	return make_float4(a.x / k, a.y / k, a.z / k, 0);
}

#define pow2(x) ((x)*(x))

static __device__ float l2_float4(float4 a)
{
	return sqrtf(pow2(a.x) + pow2(a.y) + pow2(a.z));

}

__device__ FORCEINLINE float depth_convert_cu(
	const float &f, // focal length
	const Camera_cu & cam_ref, 
	const Camera_cu & cam, const float &d)
{
	float baseline = l2_float4(cam_ref.C4 - cam.C4);
	return f * baseline / d;
}

#define matvecmul4(m, v, out) \
	out->x = m[0] * v.x + m[1] * v.y + m[2] * v.z; \
	out->y = m[3] * v.x + m[4] * v.y + m[5] * v.z; \
	out->z = m[6] * v.x + m[7] * v.y + m[8] * v.z;


__device__ FORCEINLINE void get_3dpoint_cu(
	const Camera_cu & cam,
	const int2 & p,
	const float &depth,
	float4 * __restrict__ ptX)
{
	// in case camera matrix is not normalized: see page 162, 
	// then depth might not be the real depth but w and depth needs to 
	// be computed from that first
	const float4 pt = make_float4(depth * (float) p.x - cam.P_col34.x,
								  depth * (float) p.y - cam.P_col34.y,
								  depth - cam.P_col34.z,
								  0);
	matvecmul4(cam.R_inv, pt, ptX);
}

#define matvecmul4P(m, v, out) \
out->x = m[0] * v.x + m[1] * v.y + m[2] * v.z + m[3]; \
out->y = m[4] * v.x + m[5] * v.y + m[6] * v.z + m[7]; \
out->z = m[8] * v.x + m[9] * v.y + m[10] * v.z + m[11];

__device__ FORCEINLINE void project_on_camera(const float4 & X, const Camera_cu & cam, float2 * pt, float *depth)
{
	float4 tmp = make_float4(0, 0, 0, 0);
	matvecmul4P(cam.P, X, (&tmp));
	pt->x = tmp.x / tmp.z;
	pt->y = tmp.y / tmp.z;

	*depth = tmp.z;
}

/*
 * Simple and fast depth math fusion based on depth map
 */
__global__ void fusibile(GlobalState & gs, int ref_camera)
{
	int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, 
		blockIdx.y * blockDim.y + threadIdx.y);

	const int cols = gs.cameras->cols;
	const int rows = gs.cameras->rows;

	if (p.x >= cols)
		return;

	if (p.y >= rows)
		return;

	const int center = p.y * cols + p.x;
	const CameraParameters_cu & gs_cameras = *(gs.cameras);

	float4 sum_T = tex2D < float4 > (gs.color_images_textures[ref_camera],
											p.x + 0.5f, p.y + 0.5f);
	float depth = sum_T.w;
	if (depth <= 425.001) // 1.0/255.0 -- 0.0039
		return;

	float4 X;
	get_3dpoint_cu(gs_cameras.cameras[ref_camera], p, depth, &X);
	float4 sum_X = X;

	int count = 0;
	// gs.algorithm->consistent_threshold == 3
	int consistent_threshold = gs.algorithm->consistent_threshold;

	for (int i = 0; i < gs_cameras.n_cameras && count < 2*consistent_threshold; i++) {
		if (i == ref_camera)
			continue;

		// Project 3d point X on camera i
		float2 tmp_pt;
		project_on_camera(X, gs_cameras.cameras[i], &tmp_pt, &depth);

		// Boundary check
		if (tmp_pt.x < 0 || tmp_pt.x >= cols || tmp_pt.y < 0 || tmp_pt.y >= rows)
			continue;

		float4 tmp_T = tex2D < float4 > (gs.color_images_textures[i], 
			tmp_pt.x + 0.5f, tmp_pt.y + 0.5f);

		if (tmp_T.w <= 425.001) // 1.0/255.0 -- 0.0039
			continue;

		const float depth_disp = depth_convert_cu(
									gs_cameras.cameras[ref_camera].K[0], // focal_length
									gs_cameras.cameras[ref_camera],
									gs_cameras.cameras[i],
									depth);

		const float temp_disp = depth_convert_cu(
									gs_cameras.cameras[ref_camera].K[0],
									gs_cameras.cameras[ref_camera],
									gs_cameras.cameras[i],
									tmp_T.w);

		// check on depth
		if (fabsf(depth_disp - temp_disp) < gs.algorithm->depth_threshold) {
			// depth_threshold == 0.25
			float4 tmp_X;		// 3d point of consistent point on other view
			int2 tmp_p = make_int2((int) tmp_pt.x, (int) tmp_pt.y);
			get_3dpoint_cu(gs_cameras.cameras[i], tmp_p, tmp_T.w, &tmp_X);

			sum_X = sum_X + tmp_X;
			sum_T = sum_T + tmp_T;

			count++;
		}
	}

	if (count >= consistent_threshold) {
		// Average normals and points
		sum_X = sum_X/((float) count + 1.0f);
		sum_T = sum_T/((float) count + 1.0f);

		gs.pc->points[center].coord = sum_X;
		gs.pc->points[center].texture4 = sum_T;
	}
}

void dump_gpu_memory()
{
	size_t avail, total, used;
	cudaMemGetInfo(&avail, &total);

	used = total - avail;
	printf("Device memory used: %.2f MB\n", used / 1000000.0f);
}

/* Copy point cloud to global memory */
void copy_pc_to_host(GlobalState & gs, int cam, PointCloudList & pc_list)
{
	printf("Processing camera %d\n", cam);

	int height = gs.cameras->rows;
	int width = gs.cameras->cols;
	unsigned int count = pc_list.size;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			Point_cu & p = gs.pc->points[x + y * width];

			if (count == pc_list.maximum) {
				pc_list.double_resize();
			}

			if (p.coord.x != 0 && p.coord.y != 0 && p.coord.z != 0) {
				pc_list.points[count].coord = p.coord;
				pc_list.points[count].texture4[0] = p.texture4.x;
				pc_list.points[count].texture4[1] = p.texture4.y;
				pc_list.points[count].texture4[2] = p.texture4.z;
				pc_list.points[count].texture4[3] = p.texture4.w;
				count++;
			}
		}
	}
	pc_list.size = count;

	printf("Found %.2fM points\n", count / 1000000.0f);
}

void fusibile_cu(GlobalState & gs, PointCloudList & pc_list, int num_views)
{
	int rows = gs.cameras->rows;
	int cols = gs.cameras->cols;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	printf("Run gipuma\n");

	int count = 0;
	int i = 0;

	cudaGetDeviceCount(&count);
	if (count == 0) {
		fprintf(stderr, "There is no device.\n");
		return;
	}

	for (i = 0; i < count; i++) {
		cudaDeviceProp prop;
		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if (prop.major >= 1) {
				break;
			}
		}
	}
	if (i == count) {
		fprintf(stderr, "There is no device supporting CUDA.\n");
		return;
	}

	cudaSetDevice(i);
	cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024 * 128);

	dim3 grid_size;
	grid_size.x = (cols + 32 - 1) / 32;
	grid_size.y = (rows + 32 - 1) / 32;

	dim3 block_size;
	block_size.x = 32;
	block_size.y = 32;

	printf("Grid size: %d-%d block: %d-%d\n", grid_size.x, grid_size.y, block_size.x, block_size.y);

	dump_gpu_memory();

	//int shared_memory_size = sizeof(float)  * SHARED_SIZE ;
	printf("Fusing points\n");
	cudaEventRecord(start);

	for (int cam = 0; cam < num_views; cam++) {
		fusibile <<< grid_size, block_size, cam >>> (gs, cam);
		cudaDeviceSynchronize();

		copy_pc_to_host(gs, cam, pc_list);	// slower but saves memory
		cudaDeviceSynchronize();
	}

	cudaEventRecord(stop);

	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("\tELAPSED %f seconds\n", milliseconds / 1000.f);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));
}

int run_cuda(GlobalState & gs, PointCloudList & pc_list, int num_views)
{
	printf("Run cuda\n");
	fusibile_cu(gs, pc_list, num_views);

	return 0;
}
