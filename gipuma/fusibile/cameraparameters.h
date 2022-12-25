#pragma once

#include "camera.h"
#include "managed.h"
#include "config.h"

class __align__(128) CameraParameters_cu : public Managed {
public:
    Camera_cu cameras[MAX_IMAGES];
    int cols;
    int rows;
    int* viewSelectionSubset;
    int viewSelectionSubsetNumber;
    
    CameraParameters_cu() {
        cudaMallocManaged (&viewSelectionSubset, sizeof(int) * MAX_IMAGES);
    }

    ~CameraParameters_cu() {
        cudaFree (viewSelectionSubset);
    }
};
