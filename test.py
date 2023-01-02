import argparse, os, time, sys, gc, cv2, signal
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from datasets import find_dataset_def
from models import *
from utils import *
from datasets.data_io import read_pfm, save_pfm
from plyfile import PlyData, PlyElement
from PIL import Image
from gipuma import gipuma_filter
from multiprocessing import Pool
from functools import partial
import pdb

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Predict depth, filter, and fuse')
parser.add_argument('--model', default='mvsnet', help='select model')
parser.add_argument('--dataset', default='dtu_yao_eval', help='select dataset')
parser.add_argument('--testpath', help='testing data dir for some scenes')
parser.add_argument('--testpath_single_scene', help='testing data path for single scene')
parser.add_argument('--testlist', help='testing scene list')
parser.add_argument('--batch_size', type=int, default=2, help='testing batch size')
parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--outdir', default='./outputs', help='output dir')
parser.add_argument('--display', action='store_true', help='display depth images and masks')
parser.add_argument('--max_h', type=int, default=864, help='testing max h')
parser.add_argument('--max_w', type=int, default=1152, help='testing max w')
parser.add_argument('--fix_res', action='store_true', help='scene all using same res')
parser.add_argument('--num_worker', type=int, default=4, help='depth_filer worker')
parser.add_argument('--filter_method', type=str, default='normal', choices=["gipuma", "normal", "dynamic"], help="filter method")
#filter
# parser.add_argument('--conf', type=float, default=0.03, help='prob confidence')
#filter by gimupa
parser.add_argument('--fusibile_exe_path', type=str, default='./gipuma/fusibile/build/fusibile')
# parse arguments and check
args = parser.parse_args()
print("argv:", sys.argv[1:])
print_args(args)

if args.testpath_single_scene:
    args.testpath = os.path.dirname(args.testpath_single_scene)


def write_cam(file, cam):
    # file = 'outputs/dtu_testing/scan1/cams/00000000.txt'
    # cam = array([[[ 9.7026300e-01,  7.4798302e-03,  2.4193899e-01, -1.9102000e+02],
    #         [-1.4742900e-02,  9.9949300e-01,  2.8223399e-02,  3.2883201e+00],
    #         [-2.4160500e-01, -3.0951001e-02,  9.6988100e-01,  2.2540100e+01],
    #         [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]],

    #        [[ 2.0824778e+03,  0.0000000e+00,  5.9270764e+02,  0.0000000e+00],
    #         [ 0.0000000e+00,  2.0758896e+03,  4.4573114e+02,  0.0000000e+00],
    #         [ 0.0000000e+00,  0.0000000e+00,  1.0000000e+00,  0.0000000e+00],
    #         [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00]]],
    #       dtype=float32)
    intrinsic, extrinsic = cam[1][:3, :3], cam[0]
    intrinsic_new = np.zeros((4, 4))
    intrinsic_new[:3, :3] = intrinsic
    intrinsic = intrinsic_new

    projection_matrix = np.matmul(intrinsic, extrinsic)
    projection_matrix = projection_matrix[0:3][:] # (3, 4)

    f = open(file, "w")
    for i in range(0, 3):
        for j in range(0, 4):
            f.write(str(projection_matrix[i][j]) + ' ')
        f.write('\n')
    f.write('\n')
    f.close()


def save_depth(testlist):
    for scene in testlist:
        save_scene_depth([scene])

# run CasMVS model to save depth maps and confidence maps
def save_scene_depth(testlist):
    # dataset, dataloader
    MVSDataset = find_dataset_def(args.dataset) # dtu_yao_eval

    test_dataset = MVSDataset(args.testpath, testlist, 
                              max_h=args.max_h, max_w=args.max_w, fix_res=args.fix_res)
    TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    # model
    model = TransMVSNet()

    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict['model'], strict=True)
    # model = nn.DataParallel(model) # Multi ...

    model.cuda()
    model.eval()

    with torch.no_grad():
        for batch_idx, sample in enumerate(TestImgLoader):
            sample_cuda = tocuda(sample)
            start_time = time.time()

            # sample_cuda["imgs"].size() -- torch.Size([1, 5, 3, 864, 1152])
            # sample_cuda["proj_matrix"].keys() -- dict_keys(['stage1', 'stage2', 'stage3'])
            # (Pdb) sample_cuda["proj_matrix"]['stage1'].size() -- [1, 5, 2, 4, 4]
            # (Pdb) sample_cuda["proj_matrix"]['stage2'].size() -- [1, 5, 2, 4, 4]
            # (Pdb) sample_cuda["proj_matrix"]['stage3'].size() -- [1, 5, 2, 4, 4]
            # sample_cuda["depth_values"].size() -- torch.Size([1, 192])
            outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrix"], sample_cuda["depth_values"])
            # outputs.keys() -- dict_keys(['stage1', 'depth', 'photo_confidence', 'prob_volume', 'depth_values', 'stage2', 'stage3'])
            #  outputs['stage1'].keys() -- dict_keys(['depth', 'photo_confidence', 'prob_volume', 'depth_values'])

            end_time = time.time()
            outputs = tensor2numpy(outputs)
            del sample_cuda
            filenames = sample["filename"]
            cams = sample["proj_matrix"]["stage3"].numpy() # Orignal ?
            imgs = sample["imgs"].numpy() # Orignal ?
            print('Iter {}/{}, Time:{} Res:{}'.format(batch_idx, len(TestImgLoader), end_time - start_time, imgs[0].shape))

            # save depth maps and confidence maps
            for filename, cam, img, depth_est, photo_confidence, conf_1, conf_2 in zip(filenames, cams, imgs, \
                                    outputs["depth"], outputs["photo_confidence"],  outputs['stage1']["photo_confidence"], outputs['stage2']["photo_confidence"]):
                # filename --'scan1/{}/00000000{}'
                img = img[0]  #ref view, img.shape -- (3, 864, 1152)
                cam = cam[0]  #ref cam, cam.shape -- (2, 4, 4)
                H,W = photo_confidence.shape # (864, 1152)
                conf_1 = cv2.resize(conf_1, (W,H))
                conf_2 = cv2.resize(conf_2, (W,H))
                conf_final = photo_confidence * conf_1 * conf_2

                depth_est[conf_final < 0.01] = 0.0 # xxxx8888, ==> 'disp.dmb'
                normal_color = depth_normal(depth_est) # xxxx8888 ==> normals.dmb

                # save depth maps
                depth_filename = os.path.join(args.outdir, filename.format('depth', '.pfm'))
                os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
                # save_pfm(depth_filename, depth_est)
                depth_color = visualize_depth(depth_est, depth_min=425.0, depth_max=935.0)
                cv2.imwrite(os.path.join(args.outdir, filename.format('depth', '.png')), depth_color)

                normal_filename = os.path.join(args.outdir, filename.format('normal', '.png'))
                os.makedirs(normal_filename.rsplit('/', 1)[0], exist_ok=True)
                cv2.imwrite(normal_filename, normal_color)

                # # save confidence maps
                # confidence_filename = os.path.join(args.outdir, filename.format('confidence', '.pfm'))
                # os.makedirs(confidence_filename.rsplit('/', 1)[0], exist_ok=True)
                # save_pfm(confidence_filename, conf_final)
                # # cv2.imwrite(os.path.join(args.outdir, filename.format('confidence', '_1.png')),visualize_depth(conf_1))
                # # cv2.imwrite(os.path.join(args.outdir, filename.format('confidence', '_2.png')),visualize_depth(conf_2))
                # # cv2.imwrite(os.path.join(args.outdir, filename.format('confidence', '_3.png')), visualize_depth(photo_confidence))
                # cv2.imwrite(os.path.join(args.outdir, filename.format('confidence', '_final.png')),
                #     visualize_depth(conf_final, depth_min=0.0, depth_max=1.0))

                # save cams, img
                cam_filename = os.path.join(args.outdir, filename.format('camera', '.txt'))
                os.makedirs(cam_filename.rsplit('/', 1)[0], exist_ok=True)
                write_cam(cam_filename, cam) # cam.shape -- (2, 4, 4)

                img_filename = os.path.join(args.outdir, filename.format('image', '.png'))
                os.makedirs(img_filename.rsplit('/', 1)[0], exist_ok=True)
                img = np.clip(np.transpose(img, (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # img_bgr.shape -- (864, 1152, 3), dtype=uint8
                cv2.imwrite(img_filename, img_bgr)

    torch.cuda.empty_cache()
    gc.collect()


if __name__ == '__main__':

    if args.testlist != "all":
        with open(args.testlist) as f:
            content = f.readlines()
            testlist = [line.rstrip() for line in content]
    else:
        testlist = [e for e in os.listdir(args.testpath) if os.path.isdir(os.path.join(args.testpath, e))] \
            if not args.testpath_single_scene else [os.path.basename(args.testpath_single_scene)]

    # step1. save all the depth maps and the masks in outputs directory
    save_depth(testlist)

    # step2. filter saved depth maps with photometric confidence maps and geometric constraints
    # xxxx8888
    gipuma_filter(testlist, args.outdir, args.fusibile_exe_path)
