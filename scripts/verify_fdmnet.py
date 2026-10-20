"""Run all four official SIDD-width32 backbones forward and backward on RGB."""
import argparse
import gc
import torch
from models import build_model
from models.build import VARIANTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--height', type=int, default=33)
    parser.add_argument('--width', type=int, default=47)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if min(args.height, args.width, args.threads) < 1:
        parser.error('positive dimensions/threads required')
    torch.set_num_threads(args.threads)
    counts = {}
    print('Shared backbone: width=32, enc=[2,2,4,8], middle=12, dec=[2,2,2,2]')
    for variant in VARIANTS:
        torch.manual_seed(42)
        model = build_model(variant)
        x = torch.randn(1, 3, args.height, args.width, requires_grad=True)
        y = model(x)
        assert y.shape == x.shape and torch.isfinite(y).all()
        (y-torch.randn_like(y)).square().mean().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        for name, p in model.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
        if 'dfsm' in model.middle_blks._modules:
            assert all(p.grad.abs().sum() > 0 for p in model.middle_blks.dfsm.context_analyzer.parameters())
        counts[variant] = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'{variant}: total={counts[variant]:,}, trainable={trainable:,}, forward/backward PASS', flush=True)
        del model, x, y
        gc.collect()
    assert counts['fdmnet'] == counts['nafnet_dfsm']+counts['nafnet_msmf']-counts['nafnet']


if __name__ == '__main__':
    main()
