#pragma once
#include "managed.h"


class AlgorithmParameters : public Managed{
public:
    float depthMin; // CUDA
    float depthMax; // CUDA

    //threshold for consistency check
    float depthThresh;
    float normalThresh ;

    //how many views need to be consistent? (for update: numConsistentThresh+1)
    int numConsistentThresh;

    AlgorithmParameters(){
        //threshold for consistency check
        depthThresh = 0.5f;
        normalThresh = 0.52f;

        //how many views need to be consistent? (for update: numConsistentThresh+1)
        numConsistentThresh = 2;
    }
};
