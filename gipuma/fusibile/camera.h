#pragma once
#include "managed.h"
#include "config.h"
#include "opencv2/core/core.hpp"
using namespace cv;

//assuming that K1 = K2 = K, P1 = K [I | 0] and P2 = K [R | t])
struct Camera {
	Camera():P(Mat::eye(3, 4, CV_32F)), R(Mat::eye(3, 3, CV_32F)) {
	}

	Mat_ < float >P;
	Mat_ < float >K;
	Mat_ < float >R;
	Mat_ < float >RK_inv; // R_inverse * K_inverse
};

class Camera_cu:public Managed {
  public:
	float *P;
	float *K;
	float *R;
	float *RK_inv;
	float4 P_col34;				// P.col(3) == like t

	float4 C4;					// Camera center ?

	 Camera_cu() {
		 cudaMallocManaged(&P, sizeof(float) * 4 * 4);
		 cudaMallocManaged(&K, sizeof(float) * 4 * 4);
		 cudaMallocManaged(&R, sizeof(float) * 4 * 4);
		 cudaMallocManaged(&RK_inv, sizeof(float) * 4 * 4);
	}

	 ~Camera_cu() {
		cudaFree(P);
		cudaFree(K);
		cudaFree(R);
		cudaFree(RK_inv);
	}
};

class __align__(128) CameraParameters_cu:public Managed
{
  public:
	int n_cameras;
	int cols;
	int rows;
	Camera_cu cameras[MAX_IMAGES];

	CameraParameters_cu() {
	}

	~CameraParameters_cu() {
	}
};
