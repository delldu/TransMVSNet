#pragma once
#include "managed.h"


class AlgorithmParameters:public Managed {
  public:
	float depth_threshold;
	int consistent_threshold;	// how many views need to be consistent?

	 AlgorithmParameters() {
		depth_threshold = 0.25f;
		consistent_threshold = 3;
}};
