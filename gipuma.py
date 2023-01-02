import os
from utils import *
from struct import *
import pdb


def depth_map_fusion(mvsout_folder, fusibile_exe_path):
    cams_folder = os.path.join(mvsout_folder, 'cams')
    image_folder = os.path.join(mvsout_folder, 'images')

    cmd = fusibile_exe_path + " " + mvsout_folder
    print(cmd)
    os.system(cmd)


def gipuma_filter(testlist, outdir, fusibile_exe_path):
    # testlist = ['scan1']
    # outdir = 'outputs/dtu_testing'
    # fusibile_exe_path = 'gipuma/fusibile/build/fusibile'

    for scan in testlist:
        out_folder = os.path.join(outdir, scan)
        depth_map_fusion(out_folder, fusibile_exe_path)
