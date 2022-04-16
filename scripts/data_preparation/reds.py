# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
'''
for val set, extract the subset val-300

'''
import os
import time
from basicsr.utils.create_lmdb import create_lmdb_for_reds

def make_val_300(folder, dst):
    if not os.path.exists(dst):
        os.mkdir(dst)
    templates = '*9.*'
    cp_command = 'cp {} {}'.format(os.path.join(folder, templates), dst)
    os.system(cp_command)


def flatten_folders(folder):
    for vid in range(300):
        vidfolder_path = '{:03}'.format(vid)

        if not os.path.exists(os.path.join(folder, vidfolder_path)):
            continue

        print('working on .. {} .. {}'.format(folder, vid))
        for fid in range(100):
            src_filename = '{:08}'.format(fid)

            suffixes = ['.jpg', '.png']
            suffix = None

            for suf in suffixes:
                # print(os.path.join(folder, vidfolder_path, src_filename+suf))
                if os.path.exists(os.path.join(folder, vidfolder_path, src_filename+suf)):
                    suffix = suf
                    break
            assert suffix is not None


            src_filepath = os.path.join(folder, vidfolder_path, src_filename+suffix)
            dst_filepath = os.path.join(folder, '{}_{}{}'.format(vidfolder_path, src_filename, suffix))
            os.system('mv {} {}'.format(src_filepath, dst_filepath))
            time.sleep(0.001)
        os.system('rm -r {}'.format(os.path.join(folder, vidfolder_path)))


if __name__ == '__main__':
    flatten_folders('./datasets/REDS/train/train_blur_jpeg')
    flatten_folders('./datasets/REDS/train/train_sharp')

    # flatten_folders('./datasets/REDS/val/val_blur_jpeg')
    # flatten_folders('./datasets/REDS/val/val_sharp')
    # make_val_300('./datasets/REDS/val/val_blur_jpeg', './datasets/REDS/val/blur_300')
    # make_val_300('./datasets/REDS/val/val_sharp', './datasets/REDS/val/sharp_300')

    create_lmdb_for_reds()


