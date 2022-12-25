#pragma once
#include "managed.h"
#include <vector_types.h>

class Camera_cu : public Managed {
public:
    float* P;
    float4 P_col34;
    float* M_inv;
    float* R;
    float4 C4; // Camera center ?

    float f;

	float* K;

    Camera_cu() {
        cudaMallocManaged (&P,       sizeof(float) * 4 * 4);
        cudaMallocManaged (&M_inv,   sizeof(float) * 4 * 4);
        cudaMallocManaged (&K,       sizeof(float) * 4 * 4);
        cudaMallocManaged (&R,       sizeof(float) * 4 * 4);
    }

    ~Camera_cu() {
        cudaFree (P);
        cudaFree (M_inv);
        cudaFree (K);
        cudaFree (R);
    }
};
