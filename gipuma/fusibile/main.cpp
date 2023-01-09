#include "main.h"

static void get_subfolders(const char *dirname, vector < string > &subfolders, const char *extname)
{
	DIR *dir;
	struct dirent *ent;

	dir = opendir(dirname);
	if (dir != NULL) {
		while ((ent = readdir(dir)) != NULL) {
			char *name = ent->d_name;
			if (strcmp(name, ".") == 0 || strcmp(ent->d_name, "..") == 0)
				continue;

			if (strstr(name, extname))
				subfolders.push_back(string(dirname) + "/" + string(name));
		}
		closedir(dir);
	} else {
		printf("Cannot open directory %s\n", dirname);
		exit(EXIT_FAILURE);
	}
}

static void print_help()
{
	printf("fusibile input_folder ...\n");
}

static void add_images_to_texture(vector < Mat > &imgs, cudaTextureObject_t texs[])
{
	for (size_t i = 0; i < imgs.size(); i++) {
		int rows = imgs[i].rows;
		int cols = imgs[i].cols;

		// Create channel with floating point type
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc < float4 > ();

		// Allocate array with correct size and number of channels
		cudaArray *cuArray;
		checkCudaErrors(cudaMallocArray(&cuArray, &channelDesc, cols, rows));

		checkCudaErrors(cudaMemcpy2DToArray(cuArray,
											0,
											0,
											imgs[i].ptr < float >(),
											imgs[i].step[0], cols * sizeof(float) * 4, rows, cudaMemcpyHostToDevice));

		// Specify texture
		struct cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = cuArray;

		// Specify texture object parameters
		struct cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.addressMode[0] = cudaAddressModeWrap;
		texDesc.addressMode[1] = cudaAddressModeWrap;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.normalizedCoords = 0;

		// Create texture object
		checkCudaErrors(cudaCreateTextureObject(&(texs[i]), &resDesc, &texDesc, NULL));
	}
}

static int runFusibile(char *input_folder)
{
	GlobalState *gs;
	size_t i, n_rows, n_cols, n_filenames;
	char output_folder[256], file_name[512];

	vector < string > image_filenames;
	vector < string > camera_filenames;

	sprintf(output_folder, "%s/point", input_folder);
	mkdir(output_folder, 0777);

	snprintf(file_name, sizeof(file_name), "%s/image", input_folder);
	get_subfolders(file_name, image_filenames, ".png");
	if (image_filenames.size() < 1) {
		std::cout << "Error: NOT found images under folder '" << file_name << "'" << std::endl;
		return -1;
	}
	std::sort(image_filenames.begin(), image_filenames.end());
	n_filenames = image_filenames.size();

	snprintf(file_name, sizeof(file_name), "%s/camera", input_folder);
	get_subfolders(file_name, camera_filenames, ".txt");
	if (camera_filenames.size() < 1) {
		std::cout << "Error: NOT found camera files under folder '" << file_name << "'" << std::endl;
		return -1;
	}
	std::sort(camera_filenames.begin(), camera_filenames.end());

	if (image_filenames.size() != camera_filenames.size()) {
		std::cout << "Error: image/camera files DOES NOT match under '" << input_folder << "'" << std::endl;
		return -1;
	}

	vector < Mat_ < Vec4b >> color_images;	// png with alpha -- 4b
	for (i = 0; i < n_filenames; i++) {
		color_images.push_back(imread(image_filenames[i], IMREAD_UNCHANGED));	// IMREAD_COLOR, IMREAD_UNCHANGED
		if (color_images[i].rows == 0) {
			std::cout << "Image " << image_filenames[i] << " seems to be invalid" << std::endl;
			return -1;
		}
	}
	n_rows = color_images[0].rows;
	n_cols = color_images[0].cols;

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
	gs->pc->resize(n_rows * n_cols);

	PointCloudList pc_list;
	pc_list.resize(n_rows * n_cols);	// xxxx????
	pc_list.size = 0;

	// run gpu run
	vector < Mat > color_images_list(n_filenames);	// RGBA

	for (i = 0; i < n_filenames; i++) {
		color_images_list[i] = Mat::zeros(n_rows, n_cols, CV_32FC4);

		Mat_ < Vec4f > rgba(n_rows, n_cols, CV_32FC4);
		color_images[i].convertTo(rgba, CV_32FC4, 1.0 / 255.0);
		vector < Mat_ < float >>rgba_channels(4);
		split(rgba, rgba_channels);	// C++

		Mat_ < float >depth = rgba_channels.at(3);	// Save depth
		depth = 425.0 + 512.0 * depth;
		rgba_channels.pop_back();
		rgba_channels.push_back(depth);

		merge(rgba_channels, color_images_list[i]);
	}

	// Copy images to texture memory
	dump_gpu_memory();
	add_images_to_texture(color_images_list, gs->color_images_textures);

	dump_gpu_memory();
	runcuda(*gs, pc_list, n_filenames);

	// pc_list.size = 1024000;
	dump_gpu_memory();
	snprintf(file_name, sizeof(file_name), "%s/3d_model.ply", output_folder);
	save_point_cloud(file_name, pc_list);

	return 0;
}

int main(int argc, char **argv)
{
	if (argc < 2) {
		print_help();
		return -1;
	}

	for (int i = 1; i < argc; i++)
		runFusibile(argv[i]);

	return 0;
}
