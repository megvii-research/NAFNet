"""Preserve official NAFNet forward; append DFSM -> MSMF after middle blocks."""
from basicsr.models.archs.NAFNet_arch import NAFNet as _OfficialNAFNet
from models.dfsm import DFSM
from models.msmf import MSMF


class FDMNet(_OfficialNAFNet):
    def __init__(self, img_channel=3, width=32, middle_blk_num=12,
                 enc_blk_nums=(2, 2, 4, 8), dec_blk_nums=(2, 2, 2, 2),
                 use_dfsm=True, use_msmf=True, dfsm_options=None, msmf_options=None):
        if img_channel != 3:
            raise ValueError('RGB input/output required')
        if width < 1 or len(enc_blk_nums) != len(dec_blk_nums):
            raise ValueError('positive width and equal encoder/decoder depths required')
        if middle_blk_num < 0 or any(n < 0 for n in (*enc_blk_nums, *dec_blk_nums)):
            raise ValueError('block counts must be nonnegative')
        super().__init__(img_channel=3, width=width, middle_blk_num=middle_blk_num,
                         enc_blk_nums=enc_blk_nums, dec_blk_nums=dec_blk_nums)
        channels = width * 2 ** len(enc_blk_nums)
        if use_dfsm:
            self.middle_blks.add_module('dfsm', DFSM(channels, **(dfsm_options or {})))
        if use_msmf:
            self.middle_blks.add_module('msmf', MSMF(channels, **(msmf_options or {})))

    def forward(self, inp):
        if inp.ndim != 4 or inp.shape[1] != 3 or min(inp.shape[-2:]) < 1:
            raise ValueError('expected nonempty RGB BCHW images')
        if not inp.is_floating_point():
            raise TypeError('floating point RGB images required')
        return super().forward(inp)
