import unittest
import torch
from torch.nn import functional as F
from models import DFSM, MSMF


def effective(case, parameters):
    for p in parameters:
        case.assertIsNotNone(p.grad)
        case.assertTrue(torch.isfinite(p.grad).all())
        case.assertGreater(p.grad.abs().sum().item(), 0)


class TestDFSM(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)

    def test_shapes_finite_backward_threshold_gradients(self):
        for shape in [(2, 3, 32, 32), (1, 3, 31, 47), (2, 8, 17, 22), (1, 3, 1, 1)]:
            with self.subTest(shape=shape):
                m = DFSM(shape[1])
                x = torch.randn(shape, requires_grad=True)
                y = m(x)
                self.assertEqual(y.shape, x.shape)
                self.assertTrue(torch.isfinite(y).all())
                y.square().mean().backward()
                effective(self, [x, m.noise_log_scale, m.noise_log_slope])
                effective(self, m.context_analyzer.parameters())

    def test_ordered_complementary_masks(self):
        m = DFSM(3)
        x = torch.randn(2, 3, 31, 42)
        t = m.predict_thresholds(x)
        self.assertTrue(((t[:, 0] > 0) & (t[:, 0] < t[:, 1]) & (t[:, 1] < 1)).all())
        masks = m.band_masks(x, t)
        for mask in masks:
            self.assertTrue(((mask >= 0) & (mask <= 1)).all())
        torch.testing.assert_close(sum(masks), torch.ones_like(masks[0]))

    def test_attenuation_fft_reconstruction_and_real_output(self):
        m = DFSM(3)
        x = torch.randn(2, 3, 19, 27)
        spectrum = torch.fft.fft2(x, norm='ortho')
        masks = m.band_masks(x, m.predict_thresholds(x))
        high = spectrum*masks[2]
        mask = m.suppression_mask(high)
        self.assertTrue(((mask >= 0) & (mask <= 1)).all())
        self.assertTrue(((high*mask).abs() <= high.abs()+1e-6).all())
        restored = torch.fft.ifft2(sum(spectrum*band for band in masks), norm='ortho')
        torch.testing.assert_close(restored.real, x)
        filtered = spectrum*masks[0]+spectrum*masks[1]+high*mask
        self.assertLess(torch.fft.ifft2(filtered, norm='ortho').imag.abs().max().item(), 1e-6)

    def test_zero_and_double_gradcheck(self):
        torch.testing.assert_close(DFSM(3)(torch.zeros(1, 3, 5, 7)), torch.zeros(1, 3, 5, 7))
        m = DFSM(3).double()
        x = torch.randn(1, 3, 3, 4, dtype=torch.double, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(m, (x,), fast_mode=True))


class TestMSMF(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        torch.set_num_threads(2)

    def test_shapes_finite_backward(self):
        for shape in [(2, 3, 32, 32), (1, 3, 31, 47), (2, 8, 17, 22), (1, 3, 1, 1)]:
            with self.subTest(shape=shape):
                m = MSMF(shape[1])
                x = torch.randn(shape, requires_grad=True)
                y = m(x)
                self.assertEqual(y.shape, x.shape)
                self.assertTrue(torch.isfinite(y).all())
                y.square().mean().backward()
                effective(self, [x])
                effective(self, m.scale_estimator.parameters())
                for p in m.parameters():
                    self.assertIsNotNone(p.grad)
                    self.assertTrue(torch.isfinite(p.grad).all())

    def test_equations_9_10_and_softmax(self):
        m = MSMF(3)
        x = torch.randn(2, 3, 15, 21)
        w = m.scale_weights(x)
        self.assertTrue((w > 0).all())
        torch.testing.assert_close(w.sum(1), torch.ones(2))
        fs = [branch(x if s == 1 else F.interpolate(x, size=(15//s, 21//s), mode='area'))
              for s, branch in zip((1, 2, 4), m.branches)]
        detail = fs[0]-F.avg_pool2d(fs[0], 3, 1, 1, count_include_pad=False)
        enhanced = [fs[0]]
        for i in (1, 2):
            gate = m.gates[i-1](F.interpolate(detail, size=fs[i].shape[-2:], mode='area'))
            enhanced.append(fs[i]*(1+gate))
        expected = sum(w[:, i, None, None, None]*F.interpolate(m.align[i](enhanced[i]),
                       size=(15, 21), mode='bilinear', align_corners=False) for i in range(3))
        torch.testing.assert_close(m.fuse_features(x), expected)

    def test_background_response_attenuates_both_signs(self):
        m = MSMF(3)
        x = torch.randn(2, 3, 17, 19)
        with torch.no_grad():
            fused = m.fuse_features(x)
            m.bsu[0].weight.zero_()
            m.bsu[0].bias.fill_(-4)
            low = m(x)
            m.bsu[0].bias.fill_(4)
            high = m(x)
        self.assertTrue((high.abs() <= low.abs()+1e-7).all())
        self.assertTrue((low.abs() <= fused.abs()+1e-7).all())
        torch.testing.assert_close(high, fused*(1-torch.sigmoid(torch.tensor(4.))))
