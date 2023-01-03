/*
 * utility functions for camera geometry related stuff
 * most of them from: "Multiple View Geometry in computer vision" by Hartley and Zisserman
 */

#pragma once
#include <fstream>
// #include <iostream>
using namespace std;

Mat_<float> getColSubMat ( Mat_<float> M, int* indices, int numCols ) {
    Mat_<float> subMat = Mat::zeros ( M.rows,numCols,CV_32F );
    for ( int i = 0; i < numCols; i++ ) {
        M.col ( indices[i] ).copyTo ( subMat.col ( i ) );
    }
    return subMat;
}

// Multi View Geometry, page 163
Mat_<float> getCameraCenter ( Mat_<float> &P ) {
    Mat_<float> C = Mat::zeros ( 4,1,CV_32F );

    Mat_<float> M = Mat::zeros ( 3,3,CV_32F );

    int xIndices[] = { 1, 2, 3 };
    int yIndices[] = { 0, 2, 3 };
    int zIndices[] = { 0, 1, 3 };
    int tIndices[] = { 0, 1, 2 };

    // x coordinate
    M = getColSubMat ( P, xIndices, sizeof ( xIndices )/sizeof ( xIndices[0] ) );
    C ( 0,0 ) = ( float )determinant ( M );

    // y coordinate
    M = getColSubMat ( P, yIndices,sizeof ( yIndices )/sizeof ( yIndices[0] ) );
    C ( 1,0 ) = - ( float )determinant ( M );

    // z coordinate
    M = getColSubMat ( P, zIndices,sizeof ( zIndices )/sizeof ( zIndices[0] ) );
    C ( 2,0 ) = ( float )determinant ( M );

    // t coordinate
    M = getColSubMat ( P, tIndices,sizeof ( tIndices )/sizeof ( tIndices[0] ) );
    C ( 3,0 ) = - ( float )determinant ( M );

    return C;
}

Mat_<float> getTransformationMatrix ( Mat_<float> R, Mat_<float> t ) {
    Mat_<float> transMat = Mat::eye ( 4,4, CV_32F );
    //Mat_<float> Rt = - R * t;
    R.copyTo ( transMat ( Range ( 0,3 ),Range ( 0,3 ) ) );
    t.copyTo ( transMat ( Range ( 0,3 ),Range ( 3,4 ) ) );

    return transMat;
}

void transformCamera (Mat_<float> R, Mat_<float> t, Mat_<float> transform, 
    Camera &cam, Mat_<float> K ) {
    // create rotation translation matrix
    Mat_<float> transMat_original = getTransformationMatrix ( R, t );

    //transform
    Mat_<float> transMat_t = transMat_original * transform;

    // compute translated P (only consider upper 3x4 matrix)
    cam.P = K * transMat_t ( Range ( 0,3 ),Range ( 0,4 ) );
    // set R and t
    cam.R = transMat_t ( Range ( 0,3 ),Range ( 0,3 ) );
    cam.T = transMat_t ( Range ( 0,3 ),Range ( 3,4 ) );

    // set camera center C
    Mat_<float> C = getCameraCenter ( cam.P );
    C = C / C ( 3,0 );
    cam.C3 = Vec3f ( C ( 0,0 ),C ( 1,0 ),C ( 2,0 ) );
}

Mat_<float> scaleK ( Mat_<float> K, float scaleFactor ) {
    Mat_<float> K_scaled = K.clone();
    //scale focal length
    K_scaled ( 0,0 ) = K ( 0,0 ) / scaleFactor;
    K_scaled ( 1,1 ) = K ( 1,1 ) / scaleFactor;

    //scale center point
    K_scaled ( 0,2 ) = K ( 0,2 ) / scaleFactor;
    K_scaled ( 1,2 ) = K ( 1,2 ) / scaleFactor;

    return K_scaled;
}

void copyOpencvVecToFloat4 ( Vec3f &v, float4 *a)
{
    a->x = v(0);
    a->y = v(1);
    a->z = v(2);
}

void copyOpencvMatToFloatArray ( Mat_<float> &m, float **a)
{
    for (int pj = 0; pj < m.rows; pj++) {
        for (int pi = 0; pi < m.cols; pi++) {
            (*a)[pi + pj*m.cols] = m(pj,pi);
        }
    }
}

// Read camera parameters
static void read_camera_parameters(const string p_filename, Mat_<float> &P)
{
    int i, j;
    char line[512], *p;
    ifstream myfile;

    myfile.open(p_filename.c_str(),ifstream::in);

    for(i = 0; i < 4; i++) {
        if (myfile.eof())
            break;
        myfile.getline(line,512);
        if (strstr(line,"CONTOUR")!= NULL) {
            i--;
            continue;
        }
        for (j = 0, p = strtok( line, " " );  p;  p = strtok( NULL, " " ), j++) {
            P(i,j) = (float)atof(p);
        }
    }
    myfile.close();
}

void getCameraParameters (CameraParameters_cu &gs_cameras, 
    vector<string> camera_filenames)
{
    float scaleFactor = 1.0f;
    size_t numCameras = camera_filenames.size();
    vector<Camera> cameras;
    cameras.resize ( numCameras );

    for ( size_t i = 0; i < numCameras; i++ ) {
        read_camera_parameters(camera_filenames[i], cameras[i].P);
    }

    // decompose projection matrices into K, R and t
    vector<Mat_<float> > K ( numCameras );
    vector<Mat_<float> > R ( numCameras );
    vector<Mat_<float> > T ( numCameras );

    vector<Mat_<float> > C ( numCameras );
    vector<Mat_<float> > t ( numCameras );

    for ( size_t i = 0; i < numCameras; i++ ) {
        // P = K[R | T]
        decomposeProjectionMatrix (cameras[i].P, K[i], R[i], T[i] );

        // get 3-dimensional translation vectors and camera center (divide by augmented component)
        C[i] = T[i] ( Range ( 0,3 ),Range ( 0,1 ) ) / T[i] ( 3,0 );
        t[i] = -R[i] * C[i];
    }

    // transform projection matrices (R and t part) so that P1 = K [I | 0]
    Mat_<float> transform = Mat::eye ( 4,4,CV_32F );

    // assuming K is the same for all cameras
    Mat_<float> K0 = scaleK ( K[0], scaleFactor );

    // get focal length from calibration matrix
    gs_cameras.n_cameras = numCameras;
    for ( size_t i = 0; i < numCameras; i++ ) {
        cameras[i].K = scaleK(K[i], scaleFactor);

        transformCamera(R[i], t[i], transform, cameras[i], K0);
        cameras[i].M_inv = cameras[i].P.colRange ( 0,3 ).inv ();

        // K
        copyOpencvMatToFloatArray (cameras[i].K, &gs_cameras.cameras[i].K);

        // Copy data to cuda structure
        copyOpencvMatToFloatArray(cameras[i].P, &gs_cameras.cameras[i].P);
        copyOpencvMatToFloatArray(cameras[i].M_inv, &gs_cameras.cameras[i].M_inv);
        copyOpencvMatToFloatArray(cameras[i].K, &gs_cameras.cameras[i].K);
        copyOpencvMatToFloatArray(cameras[i].R, &gs_cameras.cameras[i].R);
        copyOpencvVecToFloat4(cameras[i].C3, &gs_cameras.cameras[i].C4);

        Mat_<float> tmp = cameras[i].P.col(3);
        gs_cameras.cameras[i].P_col34.x = tmp(0,0);
        gs_cameras.cameras[i].P_col34.y = tmp(1,0);
        gs_cameras.cameras[i].P_col34.z = tmp(2,0);
    }
}
