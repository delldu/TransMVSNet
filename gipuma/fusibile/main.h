#pragma once

#include <sys/types.h>
#include <sys/stat.h>			// mkdir
#include <dirent.h>

#include <iostream>

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"

// Includes CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_texture_types.h>

// CUDA helper functions
#include "helper_cuda.h"

#include "fusibile.h"
#include "cameraGeometryUtils.h"
#include "displayUtils.h"
#include "point_cloud_list.h"

using namespace std;
