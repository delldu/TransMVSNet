#pragma once

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/contrib/contrib.hpp"
#if CV_MAJOR_VERSION == 3
#include "opencv2/core/utility.hpp"
#endif

#include <omp.h>
#include <stdint.h>

using namespace cv;
using namespace std;

//pathes to input images (camera images, ground truth, ...)
struct InputFiles {
    InputFiles () : images_folder ( "" ), p_folder ( "" ), camera_folder ( "" ) {}
    vector<string> img_filenames; // input camera images (only filenames, path is set in images_folder), names can also be used for calibration data (e.g. for Strecha P, camera)
    string images_folder; // path to camera input images
    string p_folder; // path to camera projection matrix P (Strecha)
    string camera_folder; // path to camera calibration matrix K (Strecha)
};


struct Camera {
    Camera () : P ( Mat::eye ( 3,4,CV_32F ) ), R ( Mat::eye ( 3,3,CV_32F ) ) {}

    Mat_<float> P;
    Mat_<float> M_inv;
    Mat_<float> R;
    Mat_<float> t;
    Vec3f C3; // Camera Center (x, y, z)
    string id;
    Mat_<float> K;
};

//parameters for camera geometry setup (assuming that K1 = K2 = K, P1 = K [I | 0] and P2 = K [R | t])
struct CameraParameters {
    CameraParameters () {}
    Mat_<float> K; //if K varies from camera to camera: K and f need to be stored within Camera
    float f;
    vector<Camera> cameras;
    vector<int> viewSelectionSubset;
};
