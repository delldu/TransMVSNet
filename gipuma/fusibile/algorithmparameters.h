#pragma once
#include "managed.h"

//cost function
enum { PM_COST = 0, CENSUS_TRANSFORM = 1, ADAPTIVE_CENSUS = 2, CENSUS_SELFSIMILARITY = 3, PM_SELFSIMILARITY = 4, ADCENSUS = 5, ADCENSUS_SELFSIMILARITY = 6, SPARSE_CENSUS = 7 };

//cost combination
enum { COMB_ALL = 0, COMB_BEST_N = 1, COMB_ANGLE = 2, COMB_GOOD = 3};


class AlgorithmParameters : public Managed{
public:
    int box_hsize; // filter kernel width CUDA
    int box_vsize; // filter kernel height CUDA
    float alpha; // PM_COST weighting between color and gradient CUDA
    float gamma; // parameter for weight function (used e.g. in PM_COST) CUDA
    bool color_processing; // use color processing or not (otherwise just grayscale processing)
    float cam_scale; //used to rescale K in case of rescaled image size
    int num_img_processed; //number of images that are processed as reference images
    float depthMin; // CUDA
    float depthMax; // CUDA
    // hack XXX
    int cols;
    int rows;
    // fuse
    bool storePlyFiles;
    bool remove_black_background;

    //threshold for consistency check
    float depthThresh;
    float normalThresh ;

    //how many views need to be consistent? (for update: numConsistentThresh+1)
    int numConsistentThresh;
    bool saveTexture;

    AlgorithmParameters(){
        box_hsize           = 15; // filter kernel width CUDA
        box_vsize           = 15; // filter kernel height CUDA
        alpha             = 0.9f; // PM_COST weighting between color and gradient CUDA
        gamma             = 10.0f; // parameter for weight function (used e.g. in PM_COST) CUDA
        color_processing   = true; // use color processing or not (otherwise just grayscale processing)
        cam_scale         = 1.0f; //used to rescale K in case of rescaled image size
        num_img_processed   = 1; //number of images that are processed as reference images
        depthMin          = 2.0f; // CUDA
        depthMax          = 20.0f; // CUDA
        remove_black_background = false; // CUDA

        storePlyFiles = true;

        //threshold for consistency check
        depthThresh = 0.5f;
        normalThresh = 0.52f;

        //how many views need to be consistent? (for update: numConsistentThresh+1)
        numConsistentThresh = 2;
        saveTexture = true;
    }
};
