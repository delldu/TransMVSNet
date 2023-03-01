/*
 * utility functions for camera geometry related stuff
 * most of them from: "Multiple View Geometry in computer vision" by Hartley and Zisserman
 */

#pragma once
#include <fstream>
using namespace std;

Mat_ < float >getColSubMat(Mat_ < float >M, int *indices, int numCols)
{
	Mat_ < float >subMat = Mat::zeros(M.rows, numCols, CV_32F);
	for (int i = 0; i < numCols; i++) {
		M.col(indices[i]).copyTo(subMat.col(i));
	}
	return subMat;
}

// Multi View Geometry, page 163
Mat_ < float >getCameraCenter(Mat_ < float >&P)
{
	Mat_ < float >C = Mat::zeros(4, 1, CV_32F);

	Mat_ < float >M = Mat::zeros(3, 3, CV_32F);

	int xIndices[] = { 1, 2, 3 };
	int yIndices[] = { 0, 2, 3 };
	int zIndices[] = { 0, 1, 3 };
	int tIndices[] = { 0, 1, 2 };

	// x coordinate
	M = getColSubMat(P, xIndices, sizeof(xIndices) / sizeof(xIndices[0]));
	C(0, 0) = (float) determinant(M); // eigen::Matrix::determinant

	// y coordinate
	M = getColSubMat(P, yIndices, sizeof(yIndices) / sizeof(yIndices[0]));
	C(1, 0) = -(float) determinant(M);

	// z coordinate
	M = getColSubMat(P, zIndices, sizeof(zIndices) / sizeof(zIndices[0]));
	C(2, 0) = (float) determinant(M);

	// t coordinate
	M = getColSubMat(P, tIndices, sizeof(tIndices) / sizeof(tIndices[0]));

	C(3, 0) = -(float) determinant(M);

	return C;
}

Mat_ < float >getTransformationMatrix(Mat_ < float >R, Mat_ < float >t)
{
	Mat_ < float >transMat = Mat::eye(4, 4, CV_32F);
	R.copyTo(transMat(Range(0, 3), Range(0, 3)));
	t.copyTo(transMat(Range(0, 3), Range(3, 4)));
	// ==> [R | t]
	return transMat;
}

void copyOpencvVecToFloat4(Vec3f & v, float4 * a)
{
	a->x = v(0);
	a->y = v(1);
	a->z = v(2);
}

void copyOpencvMatToFloatArray(Mat_ < float >&m, float **a)
{
	for (int pj = 0; pj < m.rows; pj++) {
		for (int pi = 0; pi < m.cols; pi++) {
			(*a)[pi + pj * m.cols] = m(pj, pi);
		}
	}
}

// Read camera parameters
static void read_camera_parameters(const string p_filename, Mat_ < float >&P)
{
	int i, j;
	char line[512], *p;
	ifstream myfile;

	myfile.open(p_filename.c_str(), ifstream::in);

	for (i = 0; i < 4; i++) {
		if (myfile.eof())
			break;
		myfile.getline(line, 512);
		if (strstr(line, "CONTOUR") != NULL) {
			i--;
			continue;
		}
		for (j = 0, p = strtok(line, " "); p; p = strtok(NULL, " "), j++) {
			P(i, j) = (float) atof(p);
		}
	}
	myfile.close();
}

void get_camera_parameters(CameraParameters_cu & gs_cameras, 
	vector < string > camera_filenames)
{
	size_t n_cameras = camera_filenames.size();
	vector < Camera > cameras;
	cameras.resize(n_cameras);

	for (size_t i = 0; i < n_cameras; i++) {
		read_camera_parameters(camera_filenames[i], cameras[i].P);
	}

	// decompose projection matrices into K, R and t
	vector < Mat_ < float >>K(n_cameras);
	vector < Mat_ < float >>R(n_cameras);
	vector < Mat_ < float >>t(n_cameras);

	Mat_ < float > T;
	Mat_ < float > C;

	for (size_t i = 0; i < n_cameras; i++) {
		// P = K[R | t]
		decomposeProjectionMatrix(cameras[i].P, K[i], R[i], T);

		// get 3-dimensional translation vectors and camera center
		C = T (Range(0, 3), Range(0, 1)) / T(3, 0);
		t[i] = -R[i] * C;

		std::cout << "-------------------" << std::endl;
		std::cout << "P:" << cameras[i].P << std::endl;
		std::cout << "K:" << K[i] << std::endl;
		std::cout << "R:" << R[i] << std::endl;
		std::cout << "t:" << t[i] << std::endl;
	}

	// get focal length from calibration matrix
	gs_cameras.n_cameras = n_cameras;
	for (size_t i = 0; i < n_cameras; i++) {
		cameras[i].K = K[i];

		// Do what ?
		Mat_ < float >C = getCameraCenter(cameras[i].P);
		C = C / C(3, 0);
		cameras[i].C3 = Vec3f(C(0, 0), C(1, 0), C(2, 0));


		cameras[i].R_inv = cameras[i].P.colRange(0, 3).inv();

		// K
		copyOpencvMatToFloatArray(cameras[i].K, &gs_cameras.cameras[i].K);

		// Copy data to cuda structure
		copyOpencvMatToFloatArray(cameras[i].P, &gs_cameras.cameras[i].P);
		copyOpencvMatToFloatArray(cameras[i].R_inv, &gs_cameras.cameras[i].R_inv);
		copyOpencvMatToFloatArray(cameras[i].K, &gs_cameras.cameras[i].K);
		copyOpencvMatToFloatArray(cameras[i].R, &gs_cameras.cameras[i].R);
		copyOpencvVecToFloat4(cameras[i].C3, &gs_cameras.cameras[i].C4);

		Mat_ < float >tmp = cameras[i].P.col(3);
		gs_cameras.cameras[i].P_col34.x = tmp(0, 0);
		gs_cameras.cameras[i].P_col34.y = tmp(1, 0);
		gs_cameras.cameras[i].P_col34.z = tmp(2, 0);
	}
}
