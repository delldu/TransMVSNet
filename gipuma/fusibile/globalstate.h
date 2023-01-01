#pragma once

#include "algorithmparameters.h"
#include "managed.h"
#include "point_cloud.h"
#include "camera.h"

// Includes CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <builtin_types.h>

// includes, cuda
#include <vector_types.h>
#include <driver_functions.h>
#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <cuda_gl_interop.h>
#include <curand_kernel.h>

class GlobalState : public Managed {
public:
    CameraParameters_cu *cameras;
    // xxxx3333 
    #if 1
    AlgorithmParameters *params;
    #endif

    PointCloud *pc;

    cudaTextureObject_t color_images_textures[MAX_IMAGES];
    cudaTextureObject_t normal_depth_textures[MAX_IMAGES]; // first 3 values normal, fourth depth

    ~GlobalState() {
    }

};
