#pragma once
#include <string.h>				// memset()
#include <stdlib.h>				// malloc(), realloc(), free()
#include "managed.h"
#include <vector_types.h>		// float4

class Point_li {
  public:
	float4 coord;				// Point coordinate
	float texture4[4];			// Average texture color
};


class PointCloudList {
  public:
	Point_li * points;
	unsigned int size;
	unsigned int maximum;

	void resize(int n) {
		maximum = n;
		points = (Point_li *) malloc(sizeof(Point_li) * n);
		memset(points, 0, sizeof(Point_li) * n);
	} void double_resize() {
		maximum *= 2;
		points = (Point_li *) realloc(points, maximum * sizeof(Point_li));
		printf("New size of point cloud list is %d\n", maximum);
	}

	~PointCloudList() {
		free(points);
	}
};
