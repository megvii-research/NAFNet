# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
NAFSSR: Stereo Image Super-Resolution Using NAFNet

@InProceedings{Chu2022NAFSSR,
  author    = {Xiaojie Chu and Liangyu Chen and Wenqing Yu},
  title     = {NAFSSR: Stereo Image Super-Resolution Using NAFNet},
  booktitle = {CVPRW},
  year      = {2022},
}
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.models.archs.NAFNet_arch import LayerNorm2d, NAFBlock
from basicsr.models.archs.arch_util import MySequential
from basicsr.models.archs.local_arch import Local_Base

class GenerateRelations(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.scale = c ** -0.5

        self.norm_l = LayerNorm2d(c)
        self.norm_r = LayerNorm2d(c)

        self.l_proj = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

    def forward(self, lfeats, rfeats):
        B, C, H, W = lfeats.shape

        lfeats = lfeats.view(B, C, H, W)
        rfeats = rfeats.view(B, C, H, W)

        lfeats, rfeats = self.l_proj(self.norm_l(lfeats)), self.r_proj(self.norm_r(rfeats))

        x = lfeats.permute(0, 2, 3, 1) #B H W c
        y = rfeats.permute(0, 2, 1, 3) #B H c W

        z = torch.matmul(x, y)  #B H W W

        return self.scale * z

class FusionModule(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.relation_generator = GenerateRelations(c)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.l_proj = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

    def forward(self, lfeats, rfeats):
        B, C, H, W = lfeats.shape

        relations = self.relation_generator(lfeats, rfeats)  # B,  H,  W,  W

        lfeats_projected = self.l_proj(lfeats.view(B, C, H, W)).permute(0, 2, 3, 1)  # B, H, W, c
        rfeats_projected = self.r_proj(rfeats.view(B, C, H, W)).permute(0, 2, 3, 1)  # B, H, W, c

        lresidual = torch.matmul(torch.softmax(relations, dim=-1), rfeats_projected)  #B, H, W, c
        rresidual = torch.matmul(torch.softmax(relations.permute(0, 1, 3, 2), dim=-1), lfeats_projected) #B, H, W, c

        lresidual = lresidual.permute(0, 3, 1, 2).view(B, C, H, W) * self.beta
        rresidual = rresidual.permute(0, 3, 1, 2).view(B, C, H, W) * self.gamma
        return lfeats + lresidual, rfeats + rresidual

class DropPath(nn.Module):
    def __init__(self, drop_rate, module):
        super().__init__()
        self.drop_rate = drop_rate
        self.module = module

    def forward(self, *feats):
        if self.training and np.random.rand() < self.drop_rate:
            return feats

        new_feats = self.module(*feats)
        factor = 1. / (1 - self.drop_rate) if self.training else 1.

        if self.training and factor != 1.:
            new_feats = tuple([x+factor*(new_x-x) for x, new_x in zip(feats, new_feats)])
        return new_feats

class NAFBlockSR(nn.Module):
    def __init__(self, c, fusion=False,  drop_out_rate=0.):
        super().__init__()
        self.blk = NAFBlock(c, drop_out_rate=drop_out_rate)
        self.fusion = FusionModule(c) if fusion else None

    def forward(self, *feats):
        feats = tuple([self.blk(x) for x in feats])
        if self.fusion:
            feats = self.fusion(*feats)
        return feats


class NAFNetSR(nn.Module):
    def __init__(self, img_channel=3, width=16, num_blks=1, drop_path_rate=0., drop_out_rate=0., fusion_from=-1, fusion_to=-1, dual=True, up_scale=4):
        super().__init__()
        self.dual = dual
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.body = MySequential(
            *[DropPath(
                drop_path_rate, 
                NAFBlockSR(
                    width, 
                    fusion=(fusion_from <= i and i <= fusion_to), 
                    drop_out_rate=drop_out_rate
                )) for i in range(num_blks)]
        )

        self.up = nn.Sequential(
            nn.Conv2d(in_channels=width, out_channels=img_channel * up_scale**2, kernel_size=3, padding=1, stride=1, groups=1, bias=True),
            nn.PixelShuffle(up_scale)
        )
        self.up_scale = up_scale

    def forward(self, inp):
        inp_hr = F.interpolate(inp, scale_factor=self.up_scale, mode='bilinear')
        if self.dual:
            inp = inp.chunk(2, dim=1)
        else:
            inp = (inp, )
        feats = [self.intro(x) for x in inp]
        feats = self.body(*feats)
        out = torch.cat([self.up(x) for x in feats], dim=1)
        out = out + inp_hr
        return out


class NAFNetSRLocal(Local_Base, NAFNetSR):
    def __init__(self, *args, train_size=(1, 6, 64, 64), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        NAFNetSR.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)

if __name__ == '__main__':
    img_channel = 3
    num_blks = 64
    width = 96
    # num_blks = 32
    # width = 64
    # num_blks = 16
    # width = 48
    dual=True
    # fusion_from = 0
    # fusion_to = num_blks
    fusion_from = 0
    fusion_to = 1000
    droppath=0.1
    train_size = (1, 6, 30, 90)

    net = NAFNetSRLocal(up_scale=2,train_size=train_size, fast_imp=True, img_channel=img_channel, width=width, num_blks=num_blks, dual=dual,
                                                 fusion_from=fusion_from,
                                                 fusion_to=fusion_to, drop_path_rate=droppath)
    # net = NAFNetSR(img_channel=img_channel, width=width, num_blks=num_blks, dual=dual,
    #                                              fusion_from=fusion_from,
    #                                              fusion_to=fusion_to, drop_path_rate=droppath)

    c = 6 if dual else 3

    a = torch.randn((2, c, 24, 23))

    b = net(a)

    print(b.shape)

    # inp_shape = (6, 128, 128)
    
    inp_shape = (c, 64, 64)

    # inp_shape = (6, 256, 96)

    from ptflops import get_model_complexity_info
    FLOPS = 0
    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=True)

    # params = float(params[:-4])
    print(params)
    macs = float(macs[:-4]) + FLOPS / 10 ** 9

    print('mac', macs, params, 'fusion from .. to ', fusion_from, fusion_to)

    # from basicsr.models.archs.arch_util import measure_inference_speed
    # net = net.cuda()
    # data = torch.randn((1, 6, 128, 128)).cuda()
    # measure_inference_speed(net, (data,))




