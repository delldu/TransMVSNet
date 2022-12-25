/*
 * utility functions for visualization of results (disparity in color, warped output, ...)
 */

#pragma once
#include <sstream>
#include <fstream>

#if (CV_MAJOR_VERSION ==2)
#include <opencv2/contrib/contrib.hpp> // needed for applyColorMap!
#endif

#include "point_cloud.h"
#include "point_cloud_list.h"


static void storePlyFileBinaryPointCloud (char* plyFilePath, PointCloudList &pc) {
    cout << "store 3D points to ply file" << endl;

    FILE *outputPly;
    outputPly=fopen(plyFilePath,"wb");

    /*write header*/
    fprintf(outputPly, "ply\n");
    fprintf(outputPly, "format binary_little_endian 1.0\n");
    fprintf(outputPly, "element vertex %d\n",pc.size);
    fprintf(outputPly, "property float x\n");
    fprintf(outputPly, "property float y\n");
    fprintf(outputPly, "property float z\n");
    fprintf(outputPly, "property uchar red\n");
    fprintf(outputPly, "property uchar green\n");
    fprintf(outputPly, "property uchar blue\n");
    fprintf(outputPly, "end_header\n");

    // distImg = Mat::zeros(pc.rows,pc.cols,CV_32F);

    //write data
#pragma omp parallel for
    for(size_t i = 0; i < pc.size; i++) {
        const Point_li &p = pc.points[i];
//        const float4 normal = p.normal;
        float4 X = p.coord;
        const char color_r = (int)p.texture4[2];
        const char color_g = (int)p.texture4[1];
        const char color_b = (int)p.texture4[0];
        /*const int color = 127.0f;*/
        /*printf("Writing point %f %f %f\n", X.x, X.y, X.z);*/

        if(!(X.x < FLT_MAX && X.x > -FLT_MAX) || !(X.y < FLT_MAX && X.y > -FLT_MAX) || !(X.z < FLT_MAX && X.z >= -FLT_MAX)){
            X.x = 0.0f;
            X.y = 0.0f;
            X.z = 0.0f;
        }
#pragma omp critical
        {
            /*myfile << X.x << " " << X.y << " " << X.z << " " << normal.x << " " << normal.y << " " << normal.z << " " << color << " " << color << " " << color << endl;*/
            fwrite(&X.x,      sizeof(X.x), 1, outputPly);
            fwrite(&X.y,      sizeof(X.y), 1, outputPly);
            fwrite(&X.z,      sizeof(X.z), 1, outputPly);
            fwrite(&color_r,  sizeof(char), 1, outputPly);
            fwrite(&color_g,  sizeof(char), 1, outputPly);
            fwrite(&color_b,  sizeof(char), 1, outputPly);
        }

    }
    fclose(outputPly);
}
