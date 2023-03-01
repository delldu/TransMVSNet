from torch.utils.data import Dataset
import numpy as np
import os, cv2, time
from PIL import Image
from datasets.data_io import *
import pdb

s_h, s_w = 0, 0
class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, nviews=5, ndepths=192, interval_scale=1.0, **kwargs):
        super(MVSDataset, self).__init__()
        # datapath = 'data/dtu_test'
        # listfile = ['scan1']
        # kwargs = {'max_h': 864, 'max_w': 1152, 'fix_res': False}

        self.datapath = datapath
        self.listfile = listfile
        self.nviews = nviews # 5
        self.ndepths = ndepths # 192
        self.interval_scale = interval_scale
        self.max_h, self.max_w = kwargs["max_h"], kwargs["max_w"]
        self.fix_res = kwargs.get("fix_res", False)  #whether to fix the resolution of input image.
        self.fix_wh = False

        self.metas = self.build_list()
        # len(self.metas) -- 49
        # scan, ref_view, src_views
        # self.metas[0] -- ('scan1', 0, [10, 1, 9, 12, 11, 13, 2, 8, 14, 27])
        # pdb.set_trace()

    def build_list(self):
        metas = []
        scans = self.listfile

        interval_scale_dict = {}
        # scans

        for scan in scans:
            # determine the interval scale of each scene. default is 1.06
            interval_scale_dict[scan] = self.interval_scale

            pair_file = "{}/pair.txt".format(scan)
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # 10 10 2346.41 1 2036.53 9 1243.89 12 1052.87 11 1000.84 13 703.583 2 604.456 8 439.759 14 327.419 27 249.278
                    # ==> [10, 1, 9, 12, 11, 13, 2, 8, 14, 27]
                    # filter by no src view and fill to nviews
                    if len(src_views) > 0:
                        if len(src_views) < self.nviews: # self.nviews -- 5
                            print("{}< num_views:{}".format(len(src_views), self.nviews))
                            src_views += [src_views[0]] * (self.nviews - len(src_views))
                        metas.append((scan, ref_view, src_views))

        self.interval_scale = interval_scale_dict
        print("dataset", "metas:", len(metas), "interval_scale:{}".format(self.interval_scale))
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filename, interval_scale):
        # extrinsic
        # 0.970263 0.00747983 0.241939 -191.02
        # -0.0147429 0.999493 0.0282234 3.28832
        # -0.241605 -0.030951 0.969881 22.5401
        # 0.0 0.0 0.0 1.0

        # intrinsic
        # 2892.33 0 823.205
        # 0 2883.18 619.071
        # 0 0 1

        # 425(depth_interval) 2.5(depth_min)
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        intrinsics[:2, :] /= 4.0

        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0]) # ==> 425.0
        depth_interval = float(lines[11].split()[1]) # ==> 2.5

        if len(lines[11].split()) >= 3: # False
            num_depth = lines[11].split()[2]
            depth_max = depth_min + int(float(num_depth)) * depth_interval
            depth_interval = (depth_max - depth_min) / self.ndepths #  self.ndepths -- 192

        depth_interval *= interval_scale # 2.5

        return intrinsics, extrinsics, depth_min, depth_interval

    def read_img(self, filename):
        img = Image.open(filename)
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.

        return np_img

    def read_depth(self, filename):
        # read pfm depth file
        pdb.set_trace()

        return np.array(read_pfm(filename)[0], dtype=np.float32)

    def scale_mvs_input(self, img, intrinsics, max_w, max_h, base=32):
        h, w = img.shape[:2]
        if h > max_h or w > max_w:
            scale = 1.0 * max_h / h
            if scale * w > max_w:
                scale = 1.0 * max_w / w
            new_w, new_h = scale * w // base * base, scale * h // base * base
        else:
            new_w, new_h = 1.0 * w // base * base, 1.0 * h // base * base

        scale_w = 1.0 * new_w / w
        scale_h = 1.0 * new_h / h
        intrinsics[0, :] *= scale_w
        intrinsics[1, :] *= scale_h

        img = cv2.resize(img, (int(new_w), int(new_h)))

        return img, intrinsics

    def __getitem__(self, idx):
        global s_h, s_w
        meta = self.metas[idx]
        scan, ref_view, src_views = meta
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]
        # ==> view_ids -- [0, 10, 1, 9, 12]

        imgs = []
        depth_values = None
        proj_matrix = []

        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.datapath, '{}/images_post/{:0>8}.jpg'.format(scan, vid))
            # ==> img_filename -- 'data/dtu_test/scan24/images_post/00000000.jpg'

            if not os.path.exists(img_filename): # True
                img_filename = os.path.join(self.datapath, '{}/images/{:0>8}.jpg'.format(scan, vid))

            proj_mat_filename = os.path.join(self.datapath, '{}/cams/{:0>8}_cam.txt'.format(scan, vid))

            img = self.read_img(img_filename)
            intrinsics, extrinsics, depth_min, depth_interval = \
                self.read_cam_file(proj_mat_filename, interval_scale=self.interval_scale[scan])

            # scale input
            # self.max_w, self.max_h -- (1152, 864)
            img, intrinsics = self.scale_mvs_input(img, intrinsics, self.max_w, self.max_h)

            if self.fix_res: # False
                # using the same standard height or width in entire scene.
                s_h, s_w = img.shape[:2]
                self.fix_res = False
                self.fix_wh = True

            if i == 0:
                if not self.fix_wh: # True
                    # using the same standard height or width in each nviews.
                    s_h, s_w = img.shape[:2]

            # resize to standard height or width
            c_h, c_w = img.shape[:2]
            if (c_h != s_h) or (c_w != s_w):
                scale_h = 1.0 * s_h / c_h
                scale_w = 1.0 * s_w / c_w
                img = cv2.resize(img, (s_w, s_h))
                intrinsics[0, :] *= scale_w
                intrinsics[1, :] *= scale_h


            imgs.append(img)
            # extrinsics, intrinsics
            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)  #
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics
            proj_matrix.append(proj_mat)

            if i == 0:  # reference view
                depth_values = np.arange(depth_min, depth_interval * (self.ndepths - 0.5) + depth_min, depth_interval,
                                         dtype=np.float32)
                # ==> depth_values -- array([425. , 427.5, 430. , ... , 900. , 902.5], dtype=float32)

        #all
        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        proj_matrix = np.stack(proj_matrix)
        # ==> proj_matrix.shape -- (5, 2, 4, 4)

        stage2_pjmats = proj_matrix.copy()
        stage2_pjmats[:, 1, :2, :] = proj_matrix[:, 1, :2, :] * 2 # For K ?
        stage3_pjmats = proj_matrix.copy()
        stage3_pjmats[:, 1, :2, :] = proj_matrix[:, 1, :2, :] * 4 # For K ?

        proj_matrix_ms = {
            "stage1": proj_matrix,
            "stage2": stage2_pjmats,
            "stage3": stage3_pjmats
        }

        # imgs.shape -- (5, 3, 864, 1152)

        return {"imgs": imgs,
                "proj_matrix": proj_matrix_ms,
                "depth_values": depth_values,
                "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}"}
