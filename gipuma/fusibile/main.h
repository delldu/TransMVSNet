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
    Camera () : baseline ( 0.54f ), P ( Mat::eye ( 3,4,CV_32F ) ), R ( Mat::eye ( 3,3,CV_32F ) ), reference ( false ), depthMin ( 2.0f ), depthMax ( 20.0f ) {}
    float baseline;
    Mat_<float> P;
    Mat_<float> P_inv;
    Mat_<float> M_inv;
    //Mat_<float> K;
    Mat_<float> R;
    Mat_<float> t;
    Vec3f C;
    bool reference;
    float depthMin; //this could be figured out from the bounding volume (not done right now, but that's why this parameter is here as well and not only in AlgorithmParameters)
    float depthMax; //this could be figured out from the bounding volume (not done right now, but that's why this parameter is here as well and not only in AlgorithmParameters)
    //int id; //corresponds to the image name id (eg. 0-10), independent of order in argument list, just dependent on name
    string id;
    Mat_<float> K;
    Mat_<float> K_inv;
    //float f;
};

//parameters for camera geometry setup (assuming that K1 = K2 = K, P1 = K [I | 0] and P2 = K [R | t])
struct CameraParameters {
    CameraParameters () : rectified ( false ), idRef ( 0 ) {}
    Mat_<float> K; //if K varies from camera to camera: K and f need to be stored within Camera
    Mat_<float> K_inv; //if K varies from camera to camera: K and f need to be stored within Camera
    float f;
    bool rectified;
    vector<Camera> cameras;
    int idRef;
    vector<int> viewSelectionSubset;
};
