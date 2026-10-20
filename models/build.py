"""Shared official SIDD-width32 backbone for all four ablations."""
from copy import deepcopy

DEFAULT_BACKBONE = dict(img_channel=3, width=32, middle_blk_num=12,
                        enc_blk_nums=(2, 2, 4, 8), dec_blk_nums=(2, 2, 2, 2))
VARIANTS = {'nafnet': (False, False), 'nafnet_dfsm': (True, False),
            'nafnet_msmf': (False, True), 'fdmnet': (True, True)}


def model_configs(backbone=None):
    shared = deepcopy(DEFAULT_BACKBONE)
    if backbone:
        if set(backbone) - set(shared):
            raise ValueError('unknown backbone setting')
        shared.update(deepcopy(backbone))
    if shared['img_channel'] != 3:
        raise ValueError('all variants require RGB')
    if shared['width'] < 1 or len(shared['enc_blk_nums']) != len(shared['dec_blk_nums']):
        raise ValueError('positive width and equal encoder/decoder depths required')
    if shared['middle_blk_num'] < 0 or any(n < 0 for n in (*shared['enc_blk_nums'], *shared['dec_blk_nums'])):
        raise ValueError('block counts must be nonnegative')
    configs = {}
    for name, (dfsm, msmf) in VARIANTS.items():
        config = dict(type='NAFNet' if name == 'nafnet' else 'FDMNet', **deepcopy(shared))
        if name != 'nafnet':
            config.update(use_dfsm=dfsm, use_msmf=msmf)
        configs[name] = config
    return configs


def build_model(variant='fdmnet', backbone=None, *, dfsm_options=None, msmf_options=None):
    from basicsr.models.archs import define_network
    configs = model_configs(backbone)
    if variant not in configs:
        raise ValueError(f'unknown variant {variant!r}; choose {tuple(VARIANTS)}')
    config = configs[variant]
    if variant != 'nafnet':
        config.update(dfsm_options=deepcopy(dfsm_options), msmf_options=deepcopy(msmf_options))
    return define_network(config)
