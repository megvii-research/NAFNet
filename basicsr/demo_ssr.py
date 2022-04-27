# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import torch

# from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, tensor2img, imwrite, set_random_seed

import argparse
from basicsr.utils.options import dict2str, parse
from basicsr.utils.dist_util import get_dist_info, init_dist
import random

def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--input_l_path', type=str, required=True, help='The path to the input left image. For stereo image inference only.')
    parser.add_argument('--input_r_path', type=str, required=True, help='The path to the input right image. For stereo image inference only.')
    parser.add_argument('--output_l_path', type=str, required=True, help='The path to the output left image. For stereo image inference only.')
    parser.add_argument('--output_r_path', type=str, required=True, help='The path to the output right image. For stereo image inference only.')

    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)

    # distributed settings
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
            print('init dist .. ', args.launcher)

    opt['rank'], opt['world_size'] = get_dist_info()

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    opt['img_path'] = {
        'input_l': args.input_l_path,
        'input_r': args.input_r_path,
        'output_l': args.output_l_path,
        'output_r': args.output_r_path
    }

    return opt

def imread(img_path):
    file_client = FileClient('disk')
    img_bytes = file_client.get(img_path, None)
    try:
        img = imfrombytes(img_bytes, float32=True)
    except:
        raise Exception("path {} not working".format(img_path))

    img = img2tensor(img, bgr2rgb=True, float32=True)
    return img

def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=False)
    opt['num_gpu'] = torch.cuda.device_count()

    img_l_path = opt['img_path'].get('input_l')
    img_r_path = opt['img_path'].get('input_r')
    output_l_path = opt['img_path'].get('output_l')
    output_r_path = opt['img_path'].get('output_r')

    ## 1. read image
    img_l = imread(img_l_path)
    img_r = imread(img_r_path)
    img = torch.cat([img_l, img_r], dim=0)

    ## 2. run inference
    opt['dist'] = False
    model = create_model(opt)

    model.feed_data(data={'lq': img.unsqueeze(dim=0)})

    if model.opt['val'].get('grids', False):
        model.grids()

    model.test()

    if model.opt['val'].get('grids', False):
        model.grids_inverse()

    visuals = model.get_current_visuals()
    sr_img_l = visuals['result'][:,:3]
    sr_img_r = visuals['result'][:,3:]
    sr_img_l, sr_img_r = tensor2img([sr_img_l, sr_img_r])
    imwrite(sr_img_l, output_l_path)
    imwrite(sr_img_r, output_r_path)

    print(f'inference {img_l_path} .. finished. saved to {output_l_path}')
    print(f'inference {img_r_path} .. finished. saved to {output_r_path}')

if __name__ == '__main__':
    main()

