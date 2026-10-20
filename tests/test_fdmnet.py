import io
import unittest
import torch
from models import build_model, model_configs
from models.build import VARIANTS
from basicsr.models.archs.NAFNet_arch import NAFNet, NAFBlock
from basicsr.models.archs.FDMNet_arch import FDMNet

SMALL = dict(width=8, enc_blk_nums=(1, 1), dec_blk_nums=(1, 1), middle_blk_num=1)


class TestFDMNet(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        torch.set_num_threads(2)

    def test_four_variants_shapes_gradients_counts(self):
        counts = {}
        for variant in VARIANTS:
            for shape in [(2, 3, 32, 40), (1, 3, 31, 47), (1, 3, 1, 1), (1, 3, 7, 19)]:
                with self.subTest(variant=variant, shape=shape):
                    m = build_model(variant, SMALL)
                    counts[variant] = sum(p.numel() for p in m.parameters())
                    x = torch.randn(shape, requires_grad=True)
                    y = m(x)
                    self.assertEqual(y.shape, x.shape)
                    self.assertTrue(torch.isfinite(y).all())
                    (y-torch.randn_like(y)).square().mean().backward()
                    self.assertTrue(torch.isfinite(x.grad).all())
                    self.assertGreater(x.grad.abs().sum().item(), 0)
                    for name, p in m.named_parameters():
                        self.assertIsNotNone(p.grad, name)
                        self.assertTrue(torch.isfinite(p.grad).all(), name)
                    if 'dfsm' in m.middle_blks._modules:
                        self.assertGreater(sum(p.grad.abs().sum().item() for p in m.middle_blks.dfsm.context_analyzer.parameters()), 0)
        self.assertGreater(counts['nafnet_dfsm'], counts['nafnet'])
        self.assertGreater(counts['nafnet_msmf'], counts['nafnet'])
        self.assertEqual(counts['fdmnet'], counts['nafnet_dfsm']+counts['nafnet_msmf']-counts['nafnet'])

    def test_official_components_and_exact_disabled_equivalence(self):
        baseline = build_model('nafnet', SMALL)
        self.assertIs(type(baseline), NAFNet)
        disabled = FDMNet(**SMALL, use_dfsm=False, use_msmf=False)
        disabled.load_state_dict(baseline.state_dict(), strict=True)
        x = torch.randn(1, 3, 29, 41)
        torch.testing.assert_close(disabled(x), baseline(x), rtol=0, atol=0)
        full = build_model('fdmnet', SMALL)
        self.assertEqual(sum(isinstance(m, NAFBlock) for m in full.modules()), 5)
        baseline_shapes = {k: v.shape for k, v in baseline.state_dict().items()}
        for variant in VARIANTS:
            m = build_model(variant, SMALL)
            for name, shape in baseline_shapes.items():
                self.assertEqual(m.state_dict()[name].shape, shape)
            n = sum(p.numel() for k, p in m.named_parameters() if k in baseline_shapes)
            self.assertEqual(n, sum(p.numel() for p in baseline.parameters()))

    def test_bottleneck_order_residual_and_crop(self):
        m = build_model('fdmnet', SMALL)
        seen, handles = [], []
        for name, module in [('middle', m.middle_blks[0]), ('dfsm', m.middle_blks.dfsm),
                             ('msmf', m.middle_blks.msmf), ('up', m.ups[0])]:
            handles.append(module.register_forward_hook(lambda module, args, out, name=name: seen.append(name)))
        x = torch.randn(1, 3, 31, 47)
        m(x)
        self.assertEqual(seen, ['middle', 'dfsm', 'msmf', 'up'])
        for h in handles:
            h.remove()
        with torch.no_grad():
            m.ending.weight.zero_()
            m.ending.bias.zero_()
        torch.testing.assert_close(m(x), x, rtol=0, atol=0)

    def test_shared_configs_rgb_validation_checkpoint(self):
        for config in model_configs(SMALL).values():
            for key, value in SMALL.items():
                self.assertEqual(config[key], value)
            self.assertEqual(config['img_channel'], 3)
        for variant in VARIANTS:
            with self.assertRaises(ValueError):
                build_model(variant, {'img_channel': 1})
        with self.assertRaises(ValueError):
            build_model('unknown')
        m = build_model('fdmnet', SMALL).eval()
        buf = io.BytesIO()
        torch.save(m.state_dict(), buf)
        buf.seek(0)
        restored = build_model('fdmnet', SMALL).eval()
        restored.load_state_dict(torch.load(buf, weights_only=True), strict=True)
        x = torch.randn(1, 3, 25, 37)
        torch.testing.assert_close(m(x), restored(x), rtol=0, atol=0)
