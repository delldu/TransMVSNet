/*
 * utility functions for reading and writing files
 */

#pragma once

#include <iostream>
#include <fstream>

// Read camera parameters
static void read_camera_parameters(const string p_filename, Mat_<float> &P)
{
    int i, j;
    char line[512], *p;
    ifstream myfile;

    myfile.open(p_filename.c_str(),ifstream::in);

    for(i = 0; i < 4; i++) {
        if (myfile.eof())
            break;
        myfile.getline(line,512);
        if (strstr(line,"CONTOUR")!= NULL) {
            i--;
            continue;
        }
        for (j = 0, p = strtok( line, " " );  p;  p = strtok( NULL, " " ), j++) {
            P(i,j) = (float)atof(p);
        }
    }
    myfile.close();
}

