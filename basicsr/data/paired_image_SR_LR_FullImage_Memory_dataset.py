# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
from torch.utils import data as data
from torchvision.transforms.functional import normalize, resize

from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)
from basicsr.data.transforms import augment, paired_random_crop_hw
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding
import os
import numpy as np

import pickle


class PairedImageSRLRFullImageMemoryDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageSRLRFullImageMemoryDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        # self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None


        # data_list = []
        self.gts = None
        self.lqs = None

        self.dataroot_gt = opt['dataroot_gt']
        self.dataroot_lq = opt['dataroot_lq']



    def __getitem__(self, index):
        if self.lqs is None:
            # print('self.dataroot lq .. ', self.dataroot_lq, self.dataroot_gt)
            with open(self.dataroot_lq, 'rb') as f:
                self.lqs = pickle.load(f)
        if self.gts is None:
            with open(self.dataroot_gt, 'rb') as f:
                self.gts = pickle.load(f)
            # with open(opt['dataroot_gt'], 'rb') as f:
            #     self.gts = pickle.load(f)



        index = index % len(self.lqs)

        scale = self.opt['scale']

        # lr_id, hr_id = self.data_list[index]
        #
        # # print('lr_id, hr_id .. ', lr_id, hr_id)
        #
        # try:
        #     img_lr = np.frombuffer(self.fetcher.get(lr_id), np.uint8)
        #     img_hr = np.frombuffer(self.fetcher.get(hr_id), np.uint8)
        # except:
        #     import time
        #     # time.sleep(0.01)
        #     # raise Exception(f'nori id..{index},{lr_id},{hr_id} not working .. ')
        #     print(f'nori id..{index},{lr_id},{hr_id} not working .. ')
        #     exit(0)
        #     # return self.__getitem__(index)
        #
        #
        # h, w, c = 480, 480, 6
        #
        # if img_hr.shape[0] != h * w * c:
        #     print('index .. ', index, lr_id, hr_id, img_hr.shape, img_lr.shape)
        #
        # assert img_hr.shape[0] == h * w * c

        img_lq = self.lqs[index].copy().astype(np.float32) / 255.
        #
        # print('index .. ', index)
        # if index >= len(self.gts1):
        #     index_gt = index - len(self.gts1)
        #     img_gt = self.gts2[index_gt].copy().astype(np.float32) / 255.
        # else:
        #     index_gt = index
        #     img_gt = self.gts1[index_gt].copy().astype(np.float32) / 255.
        #
        #
        img_gt = self.gts[index].copy().astype(np.float32) / 255.

        # img_lr = img_lr.reshape(h // 4, w // 4, c).astype(np.float32) / 255.
        # img_hr = img_hr.reshape(h, w, c).astype(np.float32) / 255.

        # img_lr, img_hr = img_lr.copy(), img_hr.copy()


        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        # gt_path = self.paths[index]['gt_path']

        # gt_path_L = os.path.join(self.gt_folder, '{:04}_L.png'.format(index + 1))
        # gt_path_R = os.path.join(self.gt_folder, '{:04}_R.png'.format(index + 1))


        # print('gt path,', gt_path)
        # img_bytes = self.file_client.get(gt_path_L, 'gt')
        # try:
        #     img_gt_L = imfrombytes(img_bytes, float32=True)
        # except:
        #     raise Exception("gt path {} not working".format(gt_path_L))
        #
        # img_bytes = self.file_client.get(gt_path_R, 'gt')
        # try:
        #     img_gt_R = imfrombytes(img_bytes, float32=True)
        # except:
        #     raise Exception("gt path {} not working".format(gt_path_R))


        # lq_path_L = os.path.join(self.lq_folder, '{:04}_L.png'.format(index + 1))
        # lq_path_R = os.path.join(self.lq_folder, '{:04}_R.png'.format(index + 1))

        # lq_path = self.paths[index]['lq_path']
        # print(', lq path', lq_path)
        # img_bytes = self.file_client.get(lq_path_L, 'lq')
        # try:
        #     img_lq_L = imfrombytes(img_bytes, float32=True)
        # except:
        #     raise Exception("lq path {} not working".format(lq_path_L))

        # img_bytes = self.file_client.get(lq_path_R, 'lq')
        # try:
        #     img_lq_R = imfrombytes(img_bytes, float32=True)
        # except:
        #     raise Exception("lq path {} not working".format(lq_path_R))



        # img_gt = np.concatenate([img_gt_L, img_gt_R], axis=-1)
        # img_lq = np.concatenate([img_lq_L, img_lq_R], axis=-1)

        # img_gt = img_hr.copy()
        # img_lq = img_lr.copy()

        # augmentation for training
        rot90 = False

        if self.opt['phase'] == 'train':
            if 'gt_size_h' in self.opt and 'gt_size_w' in self.opt:
                gt_size_h = int(self.opt['gt_size_h'])
                gt_size_w = int(self.opt['gt_size_w'])
            else:
                gt_size = int(self.opt['gt_size'])
                gt_size_h, gt_size_w = gt_size, gt_size


            if 'flip_LR' in self.opt and self.opt['flip_LR']:
                if np.random.rand() < 0.5:
                    img_gt = img_gt[:, :, [3, 4, 5, 0, 1, 2]]
                    img_lq = img_lq[:, :, [3, 4, 5, 0, 1, 2]]

                    # img_gt, img_lq

            if 'flip_RGB' in self.opt and self.opt['flip_RGB']:
                idx = [
                    [0, 1, 2, 3, 4, 5],
                    [0, 2, 1, 3, 5, 4],
                    [1, 0, 2, 4, 3, 5],
                    [1, 2, 0, 4, 5, 3],
                    [2, 0, 1, 5, 3, 4],
                    [2, 1, 0, 5, 4, 3],
                ][int(np.random.rand() * 6)]

                img_gt = img_gt[:, :, idx]
                img_lq = img_lq[:, :, idx]

            if 'inverse_RGB' in self.opt and self.opt['inverse_RGB']:
                for i in range(3):
                    if np.random.rand() < 0.5:
                        img_gt[:, :, i] = 1 - img_gt[:, :, i]
                        img_gt[:, :, i+3] = 1 - img_gt[:, :, i+3]
                        img_lq[:, :, i] = 1 - img_lq[:, :, i]
                        img_lq[:, :, i+3] = 1 - img_lq[:, :, i+3]

            if 'naive_inverse_RGB' in self.opt and self.opt['naive_inverse_RGB']:
                # for i in range(3):
                if np.random.rand() < 0.5:
                    img_gt = 1 - img_gt
                    img_lq = 1 - img_lq
                    # img_gt[:, :, i] = 1 - img_gt[:, :, i]
                    # img_gt[:, :, i+3] = 1 - img_gt[:, :, i+3]
                    # img_lq[:, :, i] = 1 - img_lq[:, :, i]
                    # img_lq[:, :, i+3] = 1 - img_lq[:, :, i+3]

            if 'random_offset' in self.opt and self.opt['random_offset'] > 0:
                # if np.random.rand() < 0.9:
                S = int(self.opt['random_offset'])

                offsets = int(np.random.rand() * (S+1))  #1~S
                s2, s4 = 0, 0

                if np.random.rand() < 0.5:
                    s2 = offsets
                else:
                    s4 = offsets

                _, w, _ = img_lq.shape

                img_lq = np.concatenate([img_lq[:, s2:w-s4, :3], img_lq[:, s4:w-s2, 3:]], axis=-1)
                img_gt = np.concatenate(
                    [img_gt[:, 4 * s2:4*w-4 * s4, :3], img_gt[:, 4 * s4:4*w-4 * s2, 3:]], axis=-1)

            # random crop
            img_gt, img_lq = img_gt.copy(), img_lq.copy()
            img_gt, img_lq = paired_random_crop_hw(img_gt, img_lq, gt_size_h, gt_size_w, scale,
                                                'gt_path_L_and_R')
            # flip, rotation
            imgs, status = augment([img_gt, img_lq], self.opt['use_hflip'],
                                    self.opt['use_rot'], vflip=self.opt['use_vflip'], return_status=True)


            img_gt, img_lq = imgs
            hflip, vflip, rot90 = status

        # if self.opt['phase'] == 'train':
        #     gt_size = self.opt['gt_size']
        #     # padding
        #     img_gt, img_lq = padding(img_gt, img_lq, gt_size)
        #
        #     # random crop
        #     img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
        #                                         'gt_path_L_and_R')
        #     # flip, rotation
        #     img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'],
        #                              self.opt['use_rot'], vflip=self.opt['use_vflip'])

        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        # if scale != 1:
        #     c, h, w = img_lq.shape
        #     img_lq = resize(img_lq, [h*scale, w*scale])
            # print('img_lq .. ', img_lq.shape, img_gt.shape)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': 'lq path ',
            'gt_path': 'gt path ',
            'is_rot': 1. if rot90 else 0.
        }

    def __len__(self):
        return 3200005
        # return 1000
        # return len(self.lqs)
        # return len(self.paths)
