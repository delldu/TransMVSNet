#pragma once

#include "algorithmparameters.h"
#include "managed.h"
#include "point_cloud.h"
#include "camera.h"

class GlobalState:public Managed {
  public:
	CameraParameters_cu * cameras;
	AlgorithmParameters *algorithm;
	PointCloud *pc;

	cudaTextureObject_t color_images_textures[MAX_IMAGES];

	~GlobalState() {
}};
