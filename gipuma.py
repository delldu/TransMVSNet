import os, sys, shutil, gc
from utils import *
from datasets.data_io import read_pfm, save_pfm
from struct import *
import numpy as np
import pdb

# read intrinsics and extrinsics
def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    # TODO: assume the feature is 1/4 of the original image size
    # intrinsics[:2, :] /= 4
    return intrinsics, extrinsics

def read_gipuma_dmb(path):
    '''read Gipuma .dmb format image'''

    with open(path, "rb") as fid:
        image_type = unpack('<i', fid.read(4))[0]
        height = unpack('<i', fid.read(4))[0]
        width = unpack('<i', fid.read(4))[0]
        channel = unpack('<i', fid.read(4))[0]

        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channel), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def write_gipuma_dmb(path, image):
    '''write Gipuma .dmb format image'''

    image_shape = np.shape(image)
    width = image_shape[1]
    height = image_shape[0]
    if len(image_shape) == 3:
        channels = image_shape[2]
    else:
        channels = 1

    if len(image_shape) == 3:
        image = np.transpose(image, (2, 0, 1)).squeeze()

    with open(path, "wb") as fid:
        # fid.write(pack(1))
        fid.write(pack('<i', 1))
        fid.write(pack('<i', height))
        fid.write(pack('<i', width))
        fid.write(pack('<i', channels))
        image.tofile(fid)
    return


def mvsnet_to_gipuma_dmb(in_path, out_path):
    '''convert mvsnet .pfm output to Gipuma .dmb format'''
    # in_path = 'outputs/dtu_testing/scan1/depth_est/00000000_prob_filtered.pfm'
    # out_path = 'outputs/dtu_testing/scan1/points/2333__00000000/disp.dmb'

    image, _ = read_pfm(in_path)
    write_gipuma_dmb(out_path, image)

    return


def mvsnet_to_gipuma_cam(in_path, out_path):
    '''convert mvsnet camera to gipuma camera format'''
    # in_path = 'outputs/dtu_testing/scan1/cams/00000000_cam.txt'
    # out_path = 'outputs/dtu_testing/scan1/points/cams/00000000.jpg.P'

    intrinsic, extrinsic = read_camera_parameters(in_path)
    intrinsic_new = np.zeros((4, 4))
    intrinsic_new[:3, :3] = intrinsic

    intrinsic = intrinsic_new

    projection_matrix = np.matmul(intrinsic, extrinsic)
    projection_matrix = projection_matrix[0:3][:] # (3, 4)

    f = open(out_path, "w")
    for i in range(0, 3):
        for j in range(0, 4):
            f.write(str(projection_matrix[i][j]) + ' ')
        f.write('\n')
    f.write('\n')
    f.close()

    return


def fake_gipuma_normal(in_depth_path, out_normal_path):
    # in_depth_path = 'outputs/dtu_testing/scan1/points/2333__00000000/disp.dmb'
    # out_normal_path = 'outputs/dtu_testing/scan1/points/2333__00000000/normals.dmb'
    
    depth_image = read_gipuma_dmb(in_depth_path)
    # (Pdb) depth_image.min() -- 0.0, depth_image.max() -- 935.0
    image_shape = np.shape(depth_image) # (864, 1152)

    normal_image = np.ones_like(depth_image)
    normal_image = np.reshape(normal_image, (image_shape[0], image_shape[1], 1))
    normal_image = np.tile(normal_image, [1, 1, 3])
    normal_image = normal_image / 1.732050808
    # (Pdb) len(normal_image) -- 864
    # (Pdb) normal_image[0].shape -- (1152, 3)
    # (Pdb) normal_image[1].shape -- (1152, 3)
    # (Pdb) normal_image[863].shape -- (1152, 3)

    mask_image = np.squeeze(np.where(depth_image > 0, 1, 0))
    mask_image = np.reshape(mask_image, (image_shape[0], image_shape[1], 1))
    mask_image = np.tile(mask_image, [1, 1, 3])
    mask_image = np.float32(mask_image)
    # (Pdb) len(mask_image) -- 864
    # (Pdb) mask_image[0].shape -- (1152, 3)

    normal_image = np.multiply(normal_image, mask_image)
    normal_image = np.float32(normal_image) # (864, 1152, 3)

    write_gipuma_dmb(out_normal_path, normal_image)


def mvsnet_to_gipuma(dense_folder, gipuma_point_folder):
    # dense_folder = 'outputs/dtu_testing/scan1'
    # gipuma_point_folder = 'outputs/dtu_testing/scan1/points'

    image_folder = os.path.join(dense_folder, 'images')
    cam_folder = os.path.join(dense_folder, 'cams')

    gipuma_cam_folder = os.path.join(gipuma_point_folder, 'cams')
    gipuma_image_folder = os.path.join(gipuma_point_folder, 'images')
    if not os.path.isdir(gipuma_point_folder):
        os.mkdir(gipuma_point_folder)
    if not os.path.isdir(gipuma_cam_folder):
        os.mkdir(gipuma_cam_folder)
    if not os.path.isdir(gipuma_image_folder):
        os.mkdir(gipuma_image_folder)

    # convert cameras
    image_names = os.listdir(image_folder)
    for image_name in image_names:
        image_prefix = os.path.splitext(image_name)[0]
        in_cam_file = os.path.join(cam_folder, image_prefix + '_cam.txt')
        out_cam_file = os.path.join(gipuma_cam_folder, image_name + '.P')
        # out_cam_file = os.path.join(gipuma_cam_folder, image_name + '_cam.txt')

        # (Pdb) in_cam_file -- 'outputs/dtu_testing/scan1/cams/00000025_cam.txt'
        # (Pdb) out_cam_file -- 'outputs/dtu_testing/scan1/points/cams/00000025.jpg.P'
        mvsnet_to_gipuma_cam(in_cam_file, out_cam_file)

    # copy images to gipuma image folder
    image_names = os.listdir(image_folder)
    for image_name in image_names:
        in_image_file = os.path.join(image_folder, image_name)
        out_image_file = os.path.join(gipuma_image_folder, image_name)
        # 'outputs/dtu_testing/scan1/images/00000025.jpg' 
        # 'outputs/dtu_testing/scan1/points/images/00000025.jpg'
        shutil.copy(in_image_file, out_image_file)

        # convert depth maps and fake normal maps
    gipuma_prefix = '2333__'
    for image_name in image_names:
        image_prefix = os.path.splitext(image_name)[0]
        sub_depth_folder = os.path.join(gipuma_point_folder, gipuma_prefix + image_prefix)
        # sub_depth_folder = os.path.join(gipuma_point_folder, image_prefix)

        # 'outputs/dtu_testing/scan1/points/2333__00000025'

        if not os.path.isdir(sub_depth_folder):
            os.mkdir(sub_depth_folder)
        in_depth_pfm = os.path.join(dense_folder, "depth_est", image_prefix + '_prob_filtered.pfm')
        out_depth_dmb = os.path.join(sub_depth_folder, 'disp.dmb')
        mvsnet_to_gipuma_dmb(in_depth_pfm, out_depth_dmb)
        # in_depth_pfm, out_depth_dmb
        # 'outputs/dtu_testing/scan1/depth_est/00000025_prob_filtered.pfm'
        # 'outputs/dtu_testing/scan1/points/2333__00000025/disp.dmb'

        fake_normal_dmb = os.path.join(sub_depth_folder, 'normals.dmb')
        fake_gipuma_normal(out_depth_dmb, fake_normal_dmb)
        # out_depth_dmb, fake_normal_dmb
        # 'outputs/dtu_testing/scan1/points/2333__00000025/normals.dmb'

def probability_filter(dense_folder, prob_threshold = 0.01):
    # dense_folder = 'outputs/dtu_testing/scan1'
    # prob_threshold = 0.01
    image_folder = os.path.join(dense_folder, 'images') # 'outputs/dtu_testing/scan1/images'

    # convert cameras
    image_names = os.listdir(image_folder)
    for image_name in image_names:
        image_prefix = os.path.splitext(image_name)[0]
        init_depth_map_path = os.path.join(dense_folder, "depth_est", image_prefix + '.pfm')
        prob_map_path = os.path.join(dense_folder, "confidence", image_prefix + '.pfm')
        out_depth_map_path = os.path.join(dense_folder, "depth_est", image_prefix + '_prob_filtered.pfm')

        depth_map, _ = read_pfm(init_depth_map_path)
        prob_map, _ = read_pfm(prob_map_path)
        depth_map[prob_map < prob_threshold] = 0

        # out_depth_map_path -- 'outputs/dtu_testing/scan1/depth_est/00000000_prob_filtered.pfm'
        save_pfm(out_depth_map_path, depth_map)
        depth_color = visualize_depth(depth_map, depth_min=0.0, depth_max=1.0)
        cv2.imwrite(os.path.join(dense_folder, "depth_est", image_prefix + '_prob_filtered.png'), depth_color)


def depth_map_fusion(mvsout_folder, fusibile_exe_path):
    cams_folder = os.path.join(mvsout_folder, 'cams')
    image_folder = os.path.join(mvsout_folder, 'images')

    cmd = fusibile_exe_path + " " + mvsout_folder
    # cmd = cmd + ' -p_folder ' + cams_folder + '/'
    # cmd = cmd + ' -images_folder ' + image_folder + '/'
    # xxxx8888
    # gipuma/fusibile/build/fusibile
    #  -input_folder outputs/dtu_testing/scan1/points/ 
    #  -p_folder outputs/dtu_testing/scan1/points/cams/
    #  -images_folder outputs/dtu_testing/scan1/points/images

    print(cmd)
    os.system(cmd)


def gipuma_filter(testlist, outdir, fusibile_exe_path):
    # testlist = ['scan1']
    # outdir = 'outputs/dtu_testing'
    # fusibile_exe_path = 'gipuma/fusibile/build/fusibile'

    prob_threshold = 0.01

    for scan in testlist:

        out_folder = os.path.join(outdir, scan)

        # point_folder = os.path.join(out_folder, 'points')
        # if not os.path.isdir(point_folder):
        #     os.mkdir(point_folder)

        # # probability filter, xxxx8888
        # print('filter depth map with probability map')
        # probability_filter(out_folder, prob_threshold)
        # # ==> 'outputs/dtu_testing/scan1/depth_est/00000000_prob_filtered.pfm'

        # # # convert to gipuma format, xxxx8888
        # print('Convert mvsnet output to gipuma input')
        # mvsnet_to_gipuma(out_folder, point_folder)

        # # depth map fusion with gipuma
        # print('Run depth map fusion & filter')

        # xxxx8888
        depth_map_fusion(out_folder, fusibile_exe_path)
