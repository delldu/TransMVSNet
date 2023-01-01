/*
 * utility functions for reading and writing files
 */

#pragma once

#include <iostream>
#include <fstream>

static void readPFileStrechaPmvs(const string p_filename, Mat_<float> &P){
    ifstream myfile;
    myfile.open(p_filename.c_str(),ifstream::in);

    //cout <<"Opening file " << p_filename << endl;
    for( int i = 0; i < 4; i++){
        if (myfile.eof())
            break;
        char line[512];
        myfile.getline(line,512);
        if (strstr(line,"CONTOUR")!= NULL) {
            //printf("Skipping CONTOUR\n");
            i--;
            continue;
        }

        //cout << "Line is "<< line << endl;
        const char* p;
        int j = 0;
        for (p = strtok( line, " " );  p;  p = strtok( NULL, " " ))
        {
            float val = (float)atof(p);
            P(i,j) = val;
            j++;
        }
    }
    myfile.close();
}

static int readDmbNormal (const char *filename, Mat_<Vec3f> &img)
{
    FILE *inimage;
    inimage = fopen(filename, "rb");
    if (!inimage){
        printf("Error opening file %s",filename);
        return -1;
    }

    int32_t type, h, w, nb;

    type = -1;

    fread(&type,sizeof(int32_t), 1, inimage);
    fread(&h,sizeof(int32_t), 1, inimage);
    fread(&w,sizeof(int32_t), 1, inimage);
    fread(&nb,sizeof(int32_t), 1, inimage);

    //only support float
    if(type != 1){
        fclose(inimage);
        return -1;
    }

    int32_t dataSize = h*w*nb;

    float* data;
    data = (float*) malloc (sizeof(float)*dataSize);
    fread(data,sizeof(float), dataSize, inimage);

    img = Mat(h,w,CV_32FC3,data);

    fclose(inimage);
    return 0;

}
// read ground truth depth map file (dmb) (provided by Tola et al. "DAISY: A Fast Local Descriptor for Dense Matching" http://cvlab.epfl.ch/software/daisy)
static int readDmb(const char *filename, Mat_<float> &img)
{
    FILE *inimage;
    inimage = fopen(filename, "rb");
    if (!inimage){
        printf("Error opening file %s",filename);
        return -1;
    }

    int32_t type, h, w, nb;

    type = -1;

    fread(&type,sizeof(int32_t), 1, inimage);
    fread(&h,sizeof(int32_t), 1, inimage);
    fread(&w,sizeof(int32_t), 1, inimage);
    fread(&nb,sizeof(int32_t), 1, inimage);

    //only support float
    if(type != 1){
        fclose(inimage);
        return -1;
    }

    int32_t dataSize = h*w*nb;

    float* data;
    data = (float*) malloc (sizeof(float)*dataSize);
    fread(data,sizeof(float), dataSize, inimage);

    img = Mat(h,w,CV_32F,data);

    fclose(inimage);
    return 0;

}
