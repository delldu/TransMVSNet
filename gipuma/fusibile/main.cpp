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

struct InputData{
    string path;
    Mat_<float> depthMap;
    Mat_<Vec3b> inputImage;
    Mat_<Vec3f> normals;
};

int getCameraFromId(string id, vector<Camera> &cameras){
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
            subfolders.push_back(string(name));
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
    // # gipuma/fusibile/build/fusibile
    // #  -input_folder outputs/dtu_testing/scan1/points/ 
    // #  -p_folder outputs/dtu_testing/scan1/points/cams/
    // #  -images_folder outputs/dtu_testing/scan1/points/images
    // #  --depth_min=0.001
    // #  --depth_max=100000
    // #  --normal_thresh=360 --disp_thresh=0.25
    // #  --num_consistent=3.0


    printf ( "\nfusibile\n" );
}

/* process command line arguments
 * Input: argc, argv - command line arguments
 * Output: inputFiles, outputFiles, parameters, gt_parameters, no_display - algorithm parameters
 */
static int getParametersFromCommandLine ( int argc,
                                          char** argv,
                                          InputFiles &inputFiles,
                                          AlgorithmParameters &parameters)
{
    const char* images_input_folder_opt = "-images_folder";
    const char* p_input_folder_opt = "-p_folder";
    const char* camera_input_folder_opt = "-camera_folder";
    const char* disp_thresh_opt = "--disp_thresh=";
    const char* normal_thresh_opt = "--normal_thresh=";
    const char* num_consistent_opt = "--num_consistent=";

    //read in arguments
    for ( int i = 1; i < argc; i++ )
    {
        if ( argv[i][0] != '-' )
        {
            inputFiles.img_filenames.push_back ( argv[i] );
        } else if ( strcmp ( argv[i], images_input_folder_opt ) == 0 )
            inputFiles.images_folder = argv[++i];
        else if ( strcmp ( argv[i], p_input_folder_opt ) == 0 )
            inputFiles.p_folder = argv[++i];
        else if ( strcmp ( argv[i], camera_input_folder_opt ) == 0 )
            inputFiles.camera_folder = argv[++i];
        else if ( strncmp ( argv[i], disp_thresh_opt, strlen (disp_thresh_opt) ) == 0 )
            sscanf ( argv[i] + strlen (disp_thresh_opt), "%f", &parameters.depthThresh );
        else if ( strncmp ( argv[i], normal_thresh_opt, strlen (normal_thresh_opt) ) == 0 ) {
            float angle_degree;
            sscanf ( argv[i] + strlen (normal_thresh_opt), "%f", &angle_degree );
            parameters.normalThresh = angle_degree * M_PI / 180.0f;
        }
        else if ( strncmp ( argv[i], num_consistent_opt, strlen (num_consistent_opt) ) == 0 ) {
            sscanf ( argv[i] + strlen (num_consistent_opt), "%d", &parameters.numConsistentThresh );
            std::cout << "Debug: parameters.numConsistentThresh: " << parameters.numConsistentThresh << std::endl;
        }
        else
        {
            printf ( "Command-line parameter error: unknown option %s\n", argv[i] );
            //return -1;
        }
    }

    return 0;
}

static void selectViews ( CameraParameters &cameraParams) {
    vector<Camera> cameras = cameraParams.cameras;
    cameraParams.viewSelectionSubset.clear ();

    for ( size_t i = 0; i < cameras.size (); i++ ) {
         //select all views, dont perform selection
        cameraParams.viewSelectionSubset.push_back ( i );
    }
}

static void addImageToTextureFloatColor (vector<Mat> &imgs, cudaTextureObject_t texs[])
{
    for (size_t i=0; i<imgs.size(); i++)
    {
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

static int runFusibile (int argc, char **argv, AlgorithmParameters &algParameters)
{
    InputFiles inputFiles;
    string ext = ".png";

    string results_folder = "results/";

    const char* results_folder_opt = "-input_folder";
    const char* p_input_folder_opt = "-p_folder";
    const char* images_input_folder_opt = "-images_folder";
    for ( int i = 1; i < argc; i++ )
    {
        if ( strcmp ( argv[i], results_folder_opt ) == 0 ){
            results_folder = argv[++i]; // outputs/dtu_testing/scan1/points/
            cout << "input folder is " << results_folder << endl;
        } else if ( strcmp ( argv[i], p_input_folder_opt ) == 0 ){
            inputFiles.p_folder = argv[++i];
        } else if ( strcmp ( argv[i], images_input_folder_opt ) == 0 ){
            inputFiles.images_folder = argv[++i];
        }
    }

    cout <<"image folder is " << inputFiles.images_folder << endl;
    cout <<"p folder is " << inputFiles.p_folder << endl;


    char output_folder[256];
    sprintf(output_folder, "%s/consistencyCheck/",results_folder.c_str());
#if defined(_WIN32)
    _mkdir(output_folder);
#else
    mkdir(output_folder, 0777);
#endif

    vector<string> subfolders;
    get_subfolders(results_folder.c_str(), subfolders);
    // results_folder -- outputs/dtu_testing/scan1/points/
    std::sort(subfolders.begin(), subfolders.end());

    map< int,string> consideredIds;
    for(size_t i=0;i<subfolders.size();i++) {
        //make sure that it has the right format (DATE_TIME_INDEX)
        size_t n = std::count(subfolders[i].begin(), subfolders[i].end(), '_');
        if(n < 2)
            continue;
        if (subfolders[i][0] != '2')
            continue;

        unsigned posFirst = subfolders[i].find_first_of("_") +1;
        unsigned found = subfolders[i].substr(posFirst).find_first_of("_") + posFirst +1;
        string id_string = subfolders[i].substr(found);

        consideredIds.insert(pair<int,string>(i, id_string));

        if( access( (inputFiles.images_folder + id_string + ".png").c_str(), R_OK ) != -1 )
            inputFiles.img_filenames.push_back((id_string + ".png"));
        else if( access( (inputFiles.images_folder + id_string + ".jpg").c_str(), R_OK ) != -1 )
            inputFiles.img_filenames.push_back((id_string + ".jpg"));
        else if( access( (inputFiles.images_folder + id_string + ".ppm").c_str(), R_OK ) != -1 )
            inputFiles.img_filenames.push_back((id_string + ".ppm"));
    }
    size_t numImages = inputFiles.img_filenames.size ();
    cout << "numImages is " << numImages << endl;
    cout << "img_filenames is " << inputFiles.img_filenames.size() << endl;

    vector<Mat_<Vec3b> > img_color; // imgLeft_color, imgRight_color;
    vector<Mat_<uint8_t> > img_grayscale;
    for ( size_t i = 0; i < numImages; i++ ) {
        img_grayscale.push_back ( imread ( ( inputFiles.images_folder + inputFiles.img_filenames[i] ), IMREAD_GRAYSCALE ) );
        img_color.push_back ( imread ( ( inputFiles.images_folder + inputFiles.img_filenames[i] ), IMREAD_COLOR ) );

        if ( img_grayscale[i].rows == 0 ) {
            printf ( "Image seems to be invalid\n" );
            return -1;
        }
    }

    size_t avail;
    size_t total;
    cudaMemGetInfo( &avail, &total );
    size_t used = total - avail;
    printf("Device memory used: %fMB\n", used/1000000.0f);

    GlobalState *gs = new GlobalState;
	gs->cameras = new CameraParameters_cu;
	gs->pc = new PointCloud;
    cudaMemGetInfo( &avail, &total );
    used = total - avail;
    printf("Device memory used: %fMB\n", used/1000000.0f);

    CameraParameters camParams = getCameraParameters (*(gs->cameras), inputFiles);
    printf("Camera size is %lu\n", camParams.cameras.size());

    selectViews (camParams);
    int numSelViews = camParams.viewSelectionSubset.size ();
    cout << "Selected views: " << numSelViews << endl;
    gs->cameras->viewSelectionSubsetNumber = numSelViews;
    ofstream myfile;
    for ( int i = 0; i < numSelViews; i++ ) {
        cout << camParams.viewSelectionSubset[i] << ", ";
        gs->cameras->viewSelectionSubset[i] = camParams.viewSelectionSubset[i];
    }
    cout << endl;

    vector<InputData> inputData;

    cout << "Reading normals and depth from disk" << endl;
    cout << "Size consideredIds is " << consideredIds.size() << endl;
    for (map<int,string>::iterator it=consideredIds.begin(); it!=consideredIds.end(); ++it){
        int i = it->first;
        string id = it->second;
        int camIdx = getCameraFromId(id, camParams.cameras);
        if(camIdx < 0)// || camIdx == camParams.idRef)
            continue;

        InputData dat;
        dat.path = results_folder + subfolders[i];

        cout << "Reading image " << inputFiles.images_folder + id + ext << endl;
        dat.inputImage = imread((inputFiles.images_folder + id + ext), IMREAD_COLOR);

        //read normal
        cout << "Reading normal " << i << ": " << (dat.path + "/normals.dmb").c_str() << endl;
        readDmbNormal((dat.path + "/normals.dmb").c_str(),dat.normals);

        //read depth
        cout << "Reading disp " << i << ": " << (dat.path + "/disp.dmb").c_str() << endl;
        readDmb((dat.path + "/disp.dmb").c_str(),dat.depthMap);

        //inputData.push_back(move(dat));
        inputData.push_back(dat);
    }
    // run gpu run
    gs->params = &algParameters;

    // Init ImageInfo
    gs->cameras->cols = img_grayscale[0].cols;
    gs->cameras->rows = img_grayscale[0].rows;

    gs->pc->resize (img_grayscale[0].rows * img_grayscale[0].cols);

	PointCloudList pc_list;
    pc_list.resize (img_grayscale[0].rows * img_grayscale[0].cols); // xxxx????

    pc_list.size = 0;
    gs->pc->rows = img_grayscale[0].rows;
    gs->pc->cols = img_grayscale[0].cols;

    vector<Mat > img_color_float(img_grayscale.size());
    vector<Mat > color_images_list(img_grayscale.size());
    vector<Mat > normal_depth_list(img_grayscale.size());

    for (size_t i = 0; i<img_grayscale.size(); i++) {
        vector<Mat_<float> > rgbChannels ( 3 );
        color_images_list[i] = Mat::zeros ( img_grayscale[0].rows, img_grayscale[0].cols, CV_32FC4 );
        img_color[i].convertTo (img_color_float[i], CV_32FC3); // or CV_32F works (too)

        Mat alpha( img_grayscale[0].rows, img_grayscale[0].cols, CV_32FC1 );
        split (img_color_float[i], rgbChannels);
        rgbChannels.push_back( alpha);
        merge (rgbChannels, color_images_list[i]);

        /* Create vector of normals and disparities */
        vector<Mat_<float> > normal ( 3 );
        normal_depth_list[i] = Mat::zeros ( img_grayscale[0].rows, img_grayscale[0].cols, CV_32FC4 );
        split (inputData[i].normals, normal);
        normal.push_back( inputData[i].depthMap);
        merge (normal, normal_depth_list[i]);
    }

    // Copy images to texture memory
    addImageToTextureFloatColor (color_images_list, gs->color_images_textures);
    addImageToTextureFloatColor (normal_depth_list, gs->normal_depth_textures);

    runcuda(*gs, pc_list, numSelViews);

    char plyFile[256];
    sprintf(plyFile, "%s/final3d_model.ply", output_folder);
    printf("Writing ply file %s\n", plyFile);
    storePlyFileBinaryPointCloud (plyFile, pc_list);

    return 0;
}

int main(int argc, char **argv)
{
    if ( argc < 3 ) {
        print_help ();
        return 0;
    }

    InputFiles inputFiles;
	AlgorithmParameters* algParameters = new AlgorithmParameters;

    int ret = getParametersFromCommandLine ( argc, argv, inputFiles, *algParameters);
    if ( ret != 0 )
        return ret;

    ret = runFusibile ( argc, argv, *algParameters);

    return 0;
}

