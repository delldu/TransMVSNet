#pragma once

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#if CV_MAJOR_VERSION == 3
#include "opencv2/core/utility.hpp"
#endif

#include <omp.h>
#include <stdint.h>

using namespace cv;
using namespace std;


