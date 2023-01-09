#pragma once
#include <string.h>				// memset()
#include "managed.h"
#include <vector_types.h>		// float4

class __align__(128) Point_cu:public Managed
{
  public:
	float4 coord;				// Coordinate
	float4 texture4;			// Texture
};


class __align__(128) PointCloud:public Managed
{
  public:
	Point_cu * points;

	void resize(int n) {
		cudaMallocManaged(&points, sizeof(Point_cu) * n);
		memset(points, 0, sizeof(Point_cu) * n);
	}

	~PointCloud() {
		cudaFree(points);
	}
};
