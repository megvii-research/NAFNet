# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import (paths_from_folder,
                                    paths_from_meta_info_file,
                                    paths_from_lmdb)
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, get_root_logger

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from psf import PSF
import random
import math

logger = get_root_logger()


class RandomDegradationImageDataset(data.Dataset):
    """Single image dataset for image restoration. Using random degradation for
        HQ image to obtain LQ images.

    Read HQ (High Quality, e.g. HR (High Resolution), blurry, noisy, etc) only.

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
            dataroot_lq (str): Data root path for lq, but not using.
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
        super(RandomDegradationImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder = opt['dataroot_gt']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['gt']
            self.paths = paths_from_lmdb(self.gt_folder)
        elif 'meta_info_file' in self.opt and self.opt[
                'meta_info_file'] is not None:
            self.paths = paths_from_meta_info_file(
                self.gt_folder, self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paths_from_folder(self.gt_folder)

        self.range_deg_params = dict(
            exter_size=(80, 60),                            # 相机外部尺寸，单位mm
            N=(80, 100, 150, 180, 200, 250),                # 屏蔽网的可选目数
            # d=(40e-3, 64e-3),                               # 光栅常数d的范围，单位um -> mm
            ratio=(0.85, 0.95),                              # 光栅透光部分a相对于d的比例
            f=(30, 35, 40, 45, 50),                         # 相机的可选焦距，单位mm
            surf_size=(2/3, 1/2, 1/3, 1/4),                 # 相机可选靶面尺寸，单位英寸
            res=((1920, 1080), (3840, 2160)),               # 相机可选分辨率
            spectrum=((475e-6, 20e-6),
                      (525e-6, 20e-6), (625e-6, 20e-6)),    # BGR频谱、中心频率附近抖动范围，单位nm -> mm
            thresh=(0.85, 0.95)
        )
        self.psf = PSF(**self.random_degradation())

    def random_degradation(self):
        params = dict()
        rng = self.range_deg_params

        params["N"] = (random.choice(rng["N"]),)*2
        dx, dy = rng["exter_size"][0]/params["N"][0], rng["exter_size"][1]/params["N"][1]
        params["d"] = (round(dx, 3), round(dy, 3))
        ratio = random.uniform(rng["ratio"][0], rng["ratio"][1])
        params["a"] = (round(ratio*params["d"][0], 3), round(ratio*params["d"][1], 3))
        params["f"] = random.choice(rng["f"])
        surf_size = random.choice(rng["surf_size"])
        x, y = math.ceil(surf_size*16)*0.8, math.ceil(surf_size*16)*0.6
        params["x"] = (round(-x/2, 2), round(x/2, 2))
        params["y"] = (round(-y/2, 2), round(y/2, 2))
        params["res"] = random.choice(rng["res"])
        params["spectrum"] = tuple(round(sp[0] + random.uniform(-sp[1], sp[1]), 6)
                                   for sp in rng["spectrum"])

        return params

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]
        # print('gt path,', gt_path)
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except Exception:
            raise Exception("gt path {} not working".format(gt_path))

        # lq_path = self.paths[index]['lq_path']
        # # print(', lq path', lq_path)
        # img_bytes = self.file_client.get(lq_path, 'lq')
        # try:
        #     img_lq = imfrombytes(img_bytes, float32=True)
        # except:
        #     raise Exception("lq path {} not working".format(lq_path))

        img_lq = None
        thresh_rng = self.range_deg_params["thresh"]
        thresh = random.uniform(thresh_rng[0], thresh_rng[1])

        while True:
            try:
                params = self.random_degradation()
                self.psf.__init__(**params)
                img_lq = self.psf.blur_image(img_gt, thresh)
                break
            except Exception:
                logger.debug("Blur kernel invalid, tyr again...")

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'],
                                     self.opt['use_rot'])

        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)
