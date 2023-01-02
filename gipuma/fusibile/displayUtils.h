/*
 * utility functions for visualization of results (disparity in color, warped output, ...)
 */

#pragma once
#include <sstream>
#include <fstream>

#include "point_cloud.h"
#include "point_cloud_list.h"


static void save_point_cloud (char* ply_filename, PointCloudList &pc) {
    cout << "store 3D points to file " << ply_filename << endl;

    FILE *fp = fopen(ply_filename, "wb");

    /*write header*/
    fprintf(fp, "ply\n");
    fprintf(fp, "format binary_little_endian 1.0\n");
    fprintf(fp, "element vertex %d\n",pc.size);
    fprintf(fp, "property float x\n");
    fprintf(fp, "property float y\n");
    fprintf(fp, "property float z\n");
    fprintf(fp, "property uchar red\n");
    fprintf(fp, "property uchar green\n");
    fprintf(fp, "property uchar blue\n");
    fprintf(fp, "end_header\n");


    //write data
#pragma omp parallel for
    for(size_t i = 0; i < pc.size; i++) {
        const Point_li &p = pc.points[i];

        float4 X = p.coord;
        const char color_r = (int)p.texture4[2];
        const char color_g = (int)p.texture4[1];
        const char color_b = (int)p.texture4[0];

        if(!(X.x < FLT_MAX && X.x > -FLT_MAX) || !(X.y < FLT_MAX && X.y > -FLT_MAX) || !(X.z < FLT_MAX && X.z >= -FLT_MAX)){
            X.x = 0.0f;
            X.y = 0.0f;
            X.z = 0.0f;
        }
#pragma omp critical
        {
            /*myfile << X.x << " " << X.y << " " << X.z << " " << normal.x << " " << normal.y << " " << normal.z << " " << color << " " << color << " " << color << endl;*/
            fwrite(&X.x,      sizeof(X.x), 1, fp);
            fwrite(&X.y,      sizeof(X.y), 1, fp);
            fwrite(&X.z,      sizeof(X.z), 1, fp);
            fwrite(&color_r,  sizeof(char), 1, fp);
            fwrite(&color_g,  sizeof(char), 1, fp);
            fwrite(&color_b,  sizeof(char), 1, fp);
        }
    }
    fclose(fp);
}
