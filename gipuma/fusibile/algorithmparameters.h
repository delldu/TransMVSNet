#pragma once
#include "managed.h"


class AlgorithmParameters : public Managed{
public:
    //threshold for consistency check
    float depthThresh;
    float normalThresh ;
    int numConsistentThresh; // how many views need to be consistent?

    AlgorithmParameters(){
        //threshold for consistency check
        depthThresh = 0.25f;
        normalThresh = 0.52f;
        numConsistentThresh = 3;
    }
};
