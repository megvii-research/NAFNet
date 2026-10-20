"""DFSM, equations 1--6; see docs/implementation_assumptions.md."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class DFSM(nn.Module):
    def __init__(self, channels, hidden_channels=None, temperature=0.05):
        super().__init__()
        if channels < 1 or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('positive channels and finite positive temperature required')
        self.channels, self.temperature = channels, temperature
        hidden = hidden_channels or max(8, channels // 4)
        self.context_analyzer = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(), nn.Linear(hidden, 2))
        self.noise_log_scale = nn.Parameter(torch.full((1, channels, 1, 1), -2.0))
        self.noise_log_slope = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def predict_thresholds(self, x):
        raw = self.context_analyzer(x.mean(dim=(-2, -1)))
        low = raw[:, :1].sigmoid()
        high = low + (1 - low) * raw[:, 1:].sigmoid()
        return torch.cat((low, high), dim=1)

    def band_masks(self, x, thresholds):
        h, w = x.shape[-2:]
        fy = torch.fft.fftfreq(h, device=x.device, dtype=x.dtype)
        fx = torch.fft.fftfreq(w, device=x.device, dtype=x.dtype)
        radius = (fy[:, None].square() + fx[None, :].square()).sqrt() / math.sqrt(0.5)
        radius = radius[None, None]
        low, high = thresholds[:, 0, None, None, None], thresholds[:, 1, None, None, None]
        below_low = ((low - radius) / self.temperature).sigmoid()
        below_high = ((high - radius) / self.temperature).sigmoid()
        return below_low, below_high - below_low, 1 - below_high

    def suppression_mask(self, spectrum):
        amplitude = spectrum.abs()
        normalized = amplitude / amplitude.mean((-2, -1), keepdim=True).clamp_min(1e-6)
        sigma_n = F.softplus(self.noise_log_scale)
        slope = F.softplus(self.noise_log_slope)
        return (slope * (normalized - sigma_n)).sigmoid()

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels or min(x.shape[-2:]) < 1:
            raise ValueError('expected nonempty BCHW with configured channels')
        if not x.is_floating_point():
            raise TypeError('floating point features required')
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        low, mid, high = self.band_masks(work, self.predict_thresholds(work))
        spectrum = torch.fft.fft2(work, norm='ortho')
        high_spectrum = spectrum * high
        merged = spectrum * low + spectrum * mid + high_spectrum * self.suppression_mask(high_spectrum)
        return torch.fft.ifft2(merged, norm='ortho').real.to(x.dtype)
