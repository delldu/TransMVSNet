#include "main.h"

struct InputData
{
    string path;
    Mat_<float> depthMap;
    Mat_<Vec3b> inputImage;
    Mat_<Vec3f> normals;
};

static void get_subfolders(const char *dirname, vector<string> &subfolders)
{
    DIR *dir;
    struct dirent *ent;

    // Open directory stream
    dir = opendir (dirname);
    if (dir != NULL) {
        while ((ent = readdir (dir)) != NULL) {
            char* name = ent->d_name;
            if(strcmp(name,".") == 0 || strcmp(ent->d_name,"..") == 0)
                continue;
            subfolders.push_back(string(dirname) + "/" + string(name));
        }
        closedir (dir);
    } else {
        printf ("Cannot open directory %s\n", dirname);
        exit (EXIT_FAILURE);
    }
}

static void print_help ()
{
    printf ( "fusibile input_folder ...\n" );
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

static int runFusibile (char *input_folder)
{
    GlobalState *gs;
    size_t i, n_rows, n_cols;
    char output_folder[256], file_name[512];

    vector<string> image_filenames;
    vector<string> camera_filenames;
    vector<string> depth_filenames;
    vector<string> normal_filenames;

    sprintf(output_folder, "%s/point", input_folder);
    mkdir(output_folder, 0777);

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

    // GS
    gs = new GlobalState;
    gs->algorithm = new AlgorithmParameters;

    // GS Camera
    gs->cameras = new CameraParameters_cu;
    getCameraParameters(*(gs->cameras), camera_filenames);
    gs->cameras->cols = n_cols;
    gs->cameras->rows = n_rows;

    // GS PC
    gs->pc = new PointCloud;
    gs->pc->resize (n_rows * n_cols);

    PointCloudList pc_list;
    pc_list.resize (n_rows * n_cols); // xxxx????
    pc_list.size = 0;

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

    snprintf(file_name, sizeof(file_name), "%s/3d_model.ply", output_folder);
    save_point_cloud (file_name, pc_list);

    return 0;
}

int main(int argc, char **argv)
{
    if ( argc < 2 ) {
        print_help ();
        return -1;
    }

    for (int i = 1; i < argc; i++)
        runFusibile(argv[i]); 

    return 0;
}

