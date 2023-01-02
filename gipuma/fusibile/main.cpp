#ifdef _WIN32
#include <windows.h>
#include <ctime>
#include <direct.h>
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <sys/types.h>
#include <dirent.h>


// Includes CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_texture_types.h>
#include <vector_types.h>


#ifdef _MSC_VER
#include <io.h>
#define R_OK 04
#else
#include <unistd.h>
#endif

// CUDA helper functions
#include "helper_cuda.h"         // helper functions for CUDA error check

#include <map> // multimap

#include <sys/stat.h> // mkdir
#include <sys/types.h> // mkdir


//#include "camera.h"
#include "algorithmparameters.h"
#include "globalstate.h"
#include "fusibile.h"

#include "main.h"
#include "fileIoUtils.h"
#include "cameraGeometryUtils.h"
// #include "mathUtils.h"
#include "displayUtils.h"
#include "point_cloud_list.h"

struct InputData
{
    string path;
    Mat_<float> depthMap;
    Mat_<Vec3b> inputImage;
    Mat_<Vec3f> normals;
};

int getCameraFromId(string id, vector<Camera> &cameras)
{
    for(size_t i =0; i< cameras.size(); i++) {
        if(cameras[i].id.compare(id) == 0)
            return i;
    }
    return -1;
}

static void get_subfolders(const char *dirname, vector<string> &subfolders)
{
    DIR *dir;
    struct dirent *ent;

    // Open directory stream
    dir = opendir (dirname);
    if (dir != NULL) {
        // Print all files and directories within the directory
        while ((ent = readdir (dir)) != NULL) {
            char* name = ent->d_name;
            if(strcmp(name,".") == 0 || strcmp(ent->d_name,"..") == 0)
                continue;
            // printf ("dir %s/\n", name);
            subfolders.push_back(string(dirname) + "/" + string(name));
        }
        closedir (dir);
    } else {
        // Could not open directory
        printf ("Cannot open directory %s\n", dirname);
        exit (EXIT_FAILURE);
    }
}

static void print_help ()
{
    printf ( "fusibile input_folder\n" );
}

/* process command line arguments
 * Input: argc, argv - command line arguments
 * Output: inputFiles, outputFiles, parameters, gt_parameters, no_display - algorithm parameters
 */
#if 0
static int getParametersFromCommandLine ( int argc,
                                          char** argv,
                                          InputFiles &inputFiles)
{
    const char* images_input_folder_opt = "-images_folder";
    const char* p_input_folder_opt = "-cameras_folder";
    const char* camera_input_folder_opt = "-camera_folder";

    //read in arguments
    for ( int i = 1; i < argc; i++ )
    {
        if ( argv[i][0] != '-' )
        {
            inputFiles.img_filenames.push_back ( argv[i] );
        } else if ( strcmp ( argv[i], images_input_folder_opt ) == 0 )
            inputFiles.images_folder = argv[++i];
        else if ( strcmp ( argv[i], p_input_folder_opt ) == 0 )
            inputFiles.cameras_folder = argv[++i];
        else if ( strcmp ( argv[i], camera_input_folder_opt ) == 0 )
            inputFiles.camera_folder = argv[++i];
        else
        {
            printf ( "Command-line parameter error: unknown option %s\n", argv[i] );
            //return -1;
        }
    }

    return 0;
}
#endif

static void selectViews(CameraParameters &cameraParams)
{
    size_t i;
    vector<Camera> cameras = cameraParams.cameras;

    cameraParams.viewSelectionSubset.clear();
    for (i = 0; i < cameras.size(); i++) {
        cameraParams.viewSelectionSubset.push_back(i);
    }
}

static void addImageToTextureFloatColor(vector<Mat> &imgs, cudaTextureObject_t texs[])
{
    for (size_t i=0; i<imgs.size(); i++) {
        int rows = imgs[i].rows;
        int cols = imgs[i].cols;
        // Create channel with floating point type
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();

        // Allocate array with correct size and number of channels
        cudaArray *cuArray;
        checkCudaErrors(cudaMallocArray(&cuArray, &channelDesc, cols, rows));

        checkCudaErrors (cudaMemcpy2DToArray (cuArray,
                                              0,
                                              0,
                                              imgs[i].ptr<float>(),
                                              imgs[i].step[0],
                                              cols*sizeof(float)*4,
                                              rows,
                                              cudaMemcpyHostToDevice));

        // Specify texture
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray;

        // Specify texture object parameters
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0]   = cudaAddressModeWrap;
        texDesc.addressMode[1]   = cudaAddressModeWrap;
        texDesc.filterMode       = cudaFilterModeLinear;
        texDesc.readMode         = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        // Create texture object
        checkCudaErrors(cudaCreateTextureObject(&(texs[i]), &resDesc, &texDesc, NULL));
    }
}

static int runFusibile (char *input_folder, AlgorithmParameters &algParameters)
{
    GlobalState *gs;
    size_t i, n_rows, n_cols;
    char output_folder[256], file_name[512];

    vector<string> image_filenames;
    vector<string> camera_filenames;
    vector<string> depth_filenames;
    vector<string> normal_filenames;

    sprintf(output_folder, "%s/point/", input_folder);
#if defined(_WIN32)
    _mkdir(output_folder);
#else
    mkdir(output_folder, 0777);
#endif

    snprintf(file_name, sizeof(file_name), "%s/image", input_folder);
    get_subfolders(file_name, image_filenames);
    if (image_filenames.size() < 1) {
        std::cout << "Error: NOT find images under folder '" << file_name << "'" << std::endl;
        return -1;
    }
    std::sort(image_filenames.begin(), image_filenames.end());

    snprintf(file_name, sizeof(file_name), "%s/camera", input_folder);
    get_subfolders(file_name, camera_filenames);
    if (camera_filenames.size() < 1) {
        std::cout << "Error: NOT found camera files under folder '" << file_name << "'" << std::endl;
        return -1;
    }
    std::sort(camera_filenames.begin(), camera_filenames.end());

    snprintf(file_name, sizeof(file_name), "%s/depth", input_folder);
    get_subfolders(file_name, depth_filenames);
    if (depth_filenames.size() < 1) {
        std::cout << "Error: NOT found depth files under folder '" << file_name << "'" << std::endl;
        return -1;
    }
    std::sort(depth_filenames.begin(), depth_filenames.end());

    snprintf(file_name, sizeof(file_name), "%s/normal", input_folder);
    get_subfolders(file_name, normal_filenames);
    if (normal_filenames.size() < 1) {
        std::cout << "Error: NOT found normal files under folder '" << file_name << "'" << std::endl;
        return -1;
    }
    std::sort(normal_filenames.begin(), normal_filenames.end());

    if (image_filenames.size() != camera_filenames.size() 
        || image_filenames.size() != depth_filenames.size()
        || image_filenames.size() != normal_filenames.size()) {
        std::cout << "Error: image/camera/depth/normal files DOES NOT match under '" << input_folder << "'" << std::endl;
        return -1;
    }

    dump_gpu_memory();

    vector<Mat_<Vec3b>> image_color;
    vector<Mat_<uint8_t>> image_gray;
    for (i = 0; i < image_filenames.size(); i++) {
        image_gray.push_back(imread(image_filenames[i], IMREAD_GRAYSCALE));
        image_color.push_back(imread(image_filenames[i], IMREAD_COLOR));
        if (image_gray[i].rows == 0) {
            std::cout << "Image " << image_filenames[i] << " seems to be invalid" << std::endl;
            return -1;
        }
    }
    n_rows = image_gray[0].rows;
    n_cols = image_gray[0].cols;


    gs = new GlobalState;
    gs->cameras = new CameraParameters_cu;
    gs->pc = new PointCloud;
    CameraParameters camParams = getCameraParameters(*(gs->cameras), camera_filenames);
    selectViews(camParams); // xxxx8888
    gs->cameras->viewSelectionSubsetNumber = camera_filenames.size();
    for (i = 0; i < camera_filenames.size(); i++ ) {
        gs->cameras->viewSelectionSubset[i] = camParams.viewSelectionSubset[i];
    }

    gs->params = &algParameters;
    gs->cameras->cols = n_cols;
    gs->cameras->rows = n_rows;
    gs->pc->resize (n_rows * n_cols);

    PointCloudList pc_list;
    pc_list.resize (n_rows * n_cols); // xxxx????
    pc_list.size = 0;
    gs->pc->rows = n_rows;
    gs->pc->cols = n_cols;

    vector<InputData> inputData;
    for (i = 0; i < image_filenames.size(); i++) {
        InputData dat;

        dat.inputImage = imread(image_filenames[i], IMREAD_COLOR);

        // readDmb(depth_filenames[i], dat.depthMap);
        Mat_<uint8_t> g = imread(depth_filenames[i], IMREAD_GRAYSCALE);
        g.convertTo(dat.depthMap, CV_32FC1, 1.0/255);
        dat.depthMap = dat.depthMap * 512.0 + 425.0;

        // readDmbNormal(normal_filenames[i], dat.normals);
        Mat_<Vec3b> c = imread(normal_filenames[i], IMREAD_COLOR);
        c.convertTo(dat.normals, CV_32FC3, 1.0/255);

        inputData.push_back(dat);
    }

    // run gpu run
    vector<Mat > img_color_float(image_gray.size());
    vector<Mat > color_images_list(image_gray.size());
    vector<Mat > normal_depth_list(image_gray.size());

    for (i = 0; i < image_gray.size(); i++) {
        vector<Mat_<float> > rgbChannels (3);
        color_images_list[i] = Mat::zeros (n_rows, n_cols, CV_32FC4);
        image_color[i].convertTo(img_color_float[i], CV_32FC3); // or CV_32F works (too)

        Mat alpha( n_rows, n_cols, CV_32FC1 );
        split (img_color_float[i], rgbChannels);
        rgbChannels.push_back( alpha);
        merge (rgbChannels, color_images_list[i]);

        /* Create vector of normals and disparities */
        vector<Mat_<float> > normal ( 3 );
        normal_depth_list[i] = Mat::zeros ( n_rows, n_cols, CV_32FC4 );
        split (inputData[i].normals, normal);
        normal.push_back( inputData[i].depthMap);
        merge (normal, normal_depth_list[i]);
    }

    // Copy images to texture memory
    addImageToTextureFloatColor(color_images_list, gs->color_images_textures);
    addImageToTextureFloatColor(normal_depth_list, gs->normal_depth_textures);

    runcuda(*gs, pc_list, image_filenames.size());

    snprintf(file_name, sizeof(file_name), "%s/final3d_model.ply", output_folder);
    printf("Writing ply file %s\n", file_name);
    storePlyFileBinaryPointCloud (file_name, pc_list);

    return 0;
}

int main(int argc, char **argv)
{
    if ( argc < 2 ) {
        print_help ();
        return 0;
    }

    // InputFiles inputFiles;
	AlgorithmParameters* algParameters = new AlgorithmParameters;

#if 0
    int ret = getParametersFromCommandLine (argc, argv, inputFiles);
    if ( ret != 0 )
        return ret;
#endif
    return runFusibile (argv[1], *algParameters);
}

