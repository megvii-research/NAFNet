# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import argparse
from os import path as osp

from basicsr.utils import scandir
from basicsr.utils.lmdb_util import make_lmdb_from_imgs

def prepare_keys(folder_path, suffix='png'):
    """Prepare image path list and keys for DIV2K dataset.

    Args:
        folder_path (str): Folder path.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    img_path_list = sorted(
        list(scandir(folder_path, suffix=suffix, recursive=False)))
    keys = [img_path.split('.{}'.format(suffix))[0] for img_path in sorted(img_path_list)]

    return img_path_list, keys

def create_lmdb_for_reds():
    # folder_path = './datasets/REDS/val/sharp_300'
    # lmdb_path = './datasets/REDS/val/sharp_300.lmdb'
    # img_path_list, keys = prepare_keys(folder_path, 'png')
    # make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)
    #
    # folder_path = './datasets/REDS/val/blur_300'
    # lmdb_path = './datasets/REDS/val/blur_300.lmdb'
    # img_path_list, keys = prepare_keys(folder_path, 'jpg')
    # make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = './datasets/REDS/train/train_sharp'
    lmdb_path = './datasets/REDS/train/train_sharp.lmdb'
    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = './datasets/REDS/train/train_blur_jpeg'
    lmdb_path = './datasets/REDS/train/train_blur_jpeg.lmdb'
    img_path_list, keys = prepare_keys(folder_path, 'jpg')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)


def create_lmdb_for_gopro():
    folder_path = './datasets/GoPro/train/blur_crops'
    lmdb_path = './datasets/GoPro/train/blur_crops.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = './datasets/GoPro/train/sharp_crops'
    lmdb_path = './datasets/GoPro/train/sharp_crops.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    # folder_path = './datasets/GoPro/test/target'
    # lmdb_path = './datasets/GoPro/test/target.lmdb'

    # img_path_list, keys = prepare_keys(folder_path, 'png')
    # make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    # folder_path = './datasets/GoPro/test/input'
    # lmdb_path = './datasets/GoPro/test/input.lmdb'

    # img_path_list, keys = prepare_keys(folder_path, 'png')
    # make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

def create_lmdb_for_rain13k():
    folder_path = './datasets/Rain13k/train/input'
    lmdb_path = './datasets/Rain13k/train/input.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'jpg')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = './datasets/Rain13k/train/target'
    lmdb_path = './datasets/Rain13k/train/target.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'jpg')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

def create_lmdb_for_SIDD():
    folder_path = './datasets/SIDD/train/input_crops'
    lmdb_path = './datasets/SIDD/train/input_crops.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'PNG')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = './datasets/SIDD/train/gt_crops'
    lmdb_path = './datasets/SIDD/train/gt_crops.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'PNG')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    #for val
    '''
    
    folder_path = './datasets/SIDD/val/input_crops'
    lmdb_path = './datasets/SIDD/val/input_crops.lmdb'
    mat_path = './datasets/SIDD/ValidationNoisyBlocksSrgb.mat'
    if not osp.exists(folder_path):
        os.makedirs(folder_path)
    assert  osp.exists(mat_path)
    data = scio.loadmat(mat_path)['ValidationNoisyBlocksSrgb']
    N, B, H ,W, C = data.shape
    data = data.reshape(N*B, H, W, C)
    for i in tqdm(range(N*B)):
        cv2.imwrite(osp.join(folder_path, 'ValidationBlocksSrgb_{}.png'.format(i)), cv2.cvtColor(data[i,...], cv2.COLOR_RGB2BGR)) 
    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = './datasets/SIDD/val/gt_crops'
    lmdb_path = './datasets/SIDD/val/gt_crops.lmdb'
    mat_path = './datasets/SIDD/ValidationGtBlocksSrgb.mat'
    if not osp.exists(folder_path):
        os.makedirs(folder_path)
    assert  osp.exists(mat_path)
    data = scio.loadmat(mat_path)['ValidationGtBlocksSrgb']
    N, B, H ,W, C = data.shape
    data = data.reshape(N*B, H, W, C)
    for i in tqdm(range(N*B)):
        cv2.imwrite(osp.join(folder_path, 'ValidationBlocksSrgb_{}.png'.format(i)), cv2.cvtColor(data[i,...], cv2.COLOR_RGB2BGR)) 
    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)
    '''
