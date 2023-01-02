/* vim: ft=cpp
 * */

//#include <helper_math.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include <stdio.h>
#include "globalstate.h"
// #include "algorithmparameters.h"
#include "camera.h"
#include "config.h"

#include <vector_types.h> // float4
#include <math.h>
#include <cuda.h>
#include <curand_kernel.h>
#include "vector_operations.h"
#include "point_cloud_list.h"


#define FORCEINLINE __forceinline__

/* compute depth value from disparity or disparity value from depth
 * Input:  f         - focal length in pixel
 *         baseline  - baseline between cameras (in meters)
 *         d - either disparity or depth value
 * Output: either depth or disparity value
 */
__device__ FORCEINLINE float depth_convert_cu(
    const float &f, const Camera_cu &cam_ref, const Camera_cu &cam, const float &d )
{
    float baseline = l2_float4(cam_ref.C4 - cam.C4);
    return f * baseline / d;
}

__device__ FORCEINLINE void get_3dpoint_cu(
    const Camera_cu &cam, const int2 &p, const float &depth,
    float4 * __restrict__ ptX)
{
    // in case camera matrix is not normalized: see page 162, 
    // then depth might not be the real depth but w and depth needs to be computed from that first
    const float4 pt = make_float4 (
                                   depth * (float)p.x - cam.P_col34.x,
                                   depth * (float)p.y - cam.P_col34.y,
                                   depth - cam.P_col34.z,
                                   0);
    matvecmul4 (cam.M_inv, pt, ptX);
}

/* get angle between two vectors in 3D
 * Input: v1,v2 - vectors
 * Output: angle in radian
 */
__device__ FORCEINLINE float getAngle_cu ( const float4 &v1, const float4 &v2 ) {
    float angle = acosf ( dot4(v1, v2));
    return angle;
}

__device__ FORCEINLINE void project_on_camera (
    const float4 &X, const Camera_cu &cam, float2 *pt, float *depth)
{
    float4 tmp = make_float4 (0, 0, 0, 0);
    matvecmul4P (cam.P, X, (&tmp));
    pt->x = tmp.x / tmp.z;
    pt->y = tmp.y / tmp.z;

    *depth = tmp.z;
}

/*
 * Simple and fast depth math fusion based on depth map and normal consensus
 */
__global__ void fusibile (GlobalState &gs, int ref_camera)
{
    int2 p = make_int2 ( blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y );

    const int cols = gs.cameras->cols;
    const int rows = gs.cameras->rows;

    if (p.x >= cols)
        return;
    if (p.y >= rows)
        return;

    const int center = p.y*cols+p.x;
    const CameraParameters_cu &camParams = *(gs.cameras);

    const float4 normal = tex2D<float4> (gs.normal_depth_textures[ref_camera], p.x + 0.5f, p.y + 0.5f);
    float depth = normal.w;

    float4 X;
    get_3dpoint_cu (camParams.cameras[ref_camera], p, depth, &X);
    float4 consistent_X = X;
    float4 consistent_normal = normal;
    float4 consistent_texture4 = tex2D<float4>(gs.color_images_textures[ref_camera], p.x+0.5f, p.y+0.5f);

    int number_consistent = 0;
    for ( int i = 0; i < camParams.viewSelectionSubsetNumber; i++ ) {

        int idxCurr = camParams.viewSelectionSubset[i];
        if (idxCurr == ref_camera)
            continue;

        // Project 3d point X on camera idxCurr
        float2 tmp_pt;
        project_on_camera (X, camParams.cameras[idxCurr], &tmp_pt, &depth);

        // Boundary check
        if (tmp_pt.x >=0 && tmp_pt.x < cols && tmp_pt.y >=0 && tmp_pt.y < rows) {
            // Compute interpolated depth and normal for tmp_pt w.r.t. camera ref_camera
            float4 tmp_normal_and_depth; // first 3 components normal, fourth depth
            tmp_normal_and_depth = tex2D<float4> (gs.normal_depth_textures[idxCurr], 
                tmp_pt.x+0.5f, tmp_pt.y+0.5f);

            const float depth_disp = depth_convert_cu(
                camParams.cameras[ref_camera].K[0], 
                camParams.cameras[ref_camera], camParams.cameras[idxCurr],
                depth );
            
            const float tmp_depth_disp = depth_convert_cu(
                camParams.cameras[ref_camera].K[0],
                camParams.cameras[ref_camera], camParams.cameras[idxCurr],
                tmp_normal_and_depth.w );
            
            // First consistency check on depth
            if (fabsf(depth_disp - tmp_depth_disp) < gs.params->depthThresh) { // depthThresh == 0.25
                float angle = getAngle_cu (tmp_normal_and_depth, normal); // extract normal
                if (angle < gs.params->normalThresh) { // normalThresh == 0.52f
                    float4 tmp_X; // 3d point of consistent point on other view
                    int2 tmp_p = make_int2 ((int) tmp_pt.x, (int) tmp_pt.y);
                    get_3dpoint_cu (camParams.cameras[idxCurr], tmp_p, tmp_normal_and_depth.w, &tmp_X);

                    consistent_X = consistent_X + tmp_X;
                    consistent_normal = consistent_normal + tmp_normal_and_depth;
                    consistent_texture4 = consistent_texture4 
                        + tex2D<float4> (gs.color_images_textures[idxCurr], tmp_pt.x+0.5f, tmp_pt.y+0.5f);

                    number_consistent++;
                }
            }
        }
    }

    // Average normals and points
    consistent_X = consistent_X / ((float) number_consistent + 1.0f);
    consistent_normal = consistent_normal / ((float) number_consistent + 1.0f);
    consistent_texture4 = consistent_texture4 / ((float) number_consistent + 1.0f);

    if (number_consistent >= gs.params->numConsistentThresh) { // numConsistentThresh == 3
        gs.pc->points[center].coord  = consistent_X;
        gs.pc->points[center].normal = consistent_normal;
        gs.pc->points[center].texture4 = consistent_texture4;
    }
}

void dump_gpu_memory()
{
    size_t avail, total, used;
    cudaMemGetInfo( &avail, &total );

    used = total - avail;
    printf("Device memory used: %.2f MB\n", used/1000000.0f);    
}

/* Copy point cloud to global memory */
void copy_point_cloud_to_host(GlobalState &gs, int cam, PointCloudList &pc_list)
{
    printf("Processing camera %d\n", cam);
    unsigned int count = pc_list.size;
    for (int y=0; y<gs.pc->rows; y++) {
        for (int x=0; x<gs.pc->cols; x++) {
            Point_cu &p = gs.pc->points[x + y*gs.pc->cols];
            const float4 X = p.coord;
            const float4 normal = p.normal;
            float texture4[4];
            texture4[0] = p.texture4.x;
            texture4[1] = p.texture4.y;
            texture4[2] = p.texture4.z;
            texture4[3] = p.texture4.w;

            if (count == pc_list.maximum) {
                printf("Not enough space to save points :'(\n... allocating more! :)");
                pc_list.increase_size(pc_list.maximum*2);
            }
            if (X.x != 0 && X.y != 0 && X.z != 0) {
                pc_list.points[count].coord = X;
                pc_list.points[count].normal = normal;
                pc_list.points[count].texture4[0] = texture4[0];
                pc_list.points[count].texture4[1] = texture4[1];
                pc_list.points[count].texture4[2] = texture4[2];
                pc_list.points[count].texture4[3] = texture4[3];
                count++;
            }
            p.coord = make_float4(0,0,0,0);
        }
    }
    printf("Found %.2f million points\n", count/1000000.0f);
    pc_list.size = count;
}

template< typename T >
void fusibile_cu(GlobalState &gs, PointCloudList &pc_list, int num_views)
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
    if(count == 0) {
        fprintf(stderr, "There is no device.\n");
        return ;
    }

    for(i = 0; i < count; i++) {
        cudaDeviceProp prop;
        if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            if(prop.major >= 1) {
                break;
            }
        }
    }
    if(i == count) {
        fprintf(stderr, "There is no device supporting CUDA.\n");
        return ;
    }

    cudaSetDevice(i);
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024*128);

    dim3 grid_size_initrand;
    grid_size_initrand.x = (cols + 32-1)/32;
    grid_size_initrand.y = (rows + 32-1)/32;
    dim3 block_size_initrand;
    block_size_initrand.x = 32;
    block_size_initrand.y = 32;

    printf("Grid size initrand is grid: %d-%d block: %d-%d\n", 
        grid_size_initrand.x, grid_size_initrand.y, block_size_initrand.x, block_size_initrand.y);

    dump_gpu_memory();

    //int shared_memory_size = sizeof(float)  * SHARED_SIZE ;
    printf("Fusing points\n");
    cudaEventRecord(start);

    for (int cam=0; cam < num_views; cam++) {
        fusibile <<< grid_size_initrand, block_size_initrand, cam>>>(gs, cam);
        cudaDeviceSynchronize();

        copy_point_cloud_to_host(gs, cam, pc_list); // slower but saves memory
        cudaDeviceSynchronize();
    }

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("\t\tELAPSED %f seconds\n", milliseconds/1000.f);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
}

int runcuda(GlobalState &gs, PointCloudList &pc_list, int num_views)
{
    printf("Run cuda\n");
    fusibile_cu<float4>(gs, pc_list, num_views);
    return 0;
}
