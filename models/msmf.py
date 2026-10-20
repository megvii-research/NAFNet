"""MSMF, equations 7--12; see docs/implementation_assumptions.md."""
from torch import nn
from torch.nn import functional as F


class MSMF(nn.Module):
    def __init__(self, channels, hidden_channels=None):
        super().__init__()
        if channels < 1:
            raise ValueError('positive channels required')
        self.channels = channels
        hidden = hidden_channels or max(8, channels // 4)
        self.scale_estimator = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(), nn.Linear(hidden, 3))
        self.branches = nn.ModuleList([nn.Conv2d(channels, channels, 3, padding=1) for _ in range(3)])
        self.gates = nn.ModuleList([nn.Sequential(nn.Conv2d(channels, channels, 1), nn.Sigmoid()) for _ in range(2)])
        self.align = nn.ModuleList([nn.Conv2d(channels, channels, 1) for _ in range(3)])
        self.bsu = nn.Sequential(nn.Conv2d(channels, 1, 1), nn.Sigmoid())

    def scale_weights(self, x):
        return self.scale_estimator(x.mean((-2, -1))).softmax(dim=1)

    def fuse_features(self, x):
        h, w = x.shape[-2:]
        weights = self.scale_weights(x)
        features = []
        for scale, branch in zip((1, 2, 4), self.branches):
            source = x if scale == 1 else F.interpolate(x, size=(max(1, h//scale), max(1, w//scale)), mode='area')
            features.append(branch(source))
        detail = features[0] - F.avg_pool2d(features[0], 3, 1, 1, count_include_pad=False)
        enhanced = [features[0]]
        for feature, gate in zip(features[1:], self.gates):
            down = F.interpolate(detail, size=feature.shape[-2:], mode='area')
            enhanced.append(feature + gate(down) * feature)  # Eq. 9, not additive detail injection.
        aligned = [F.interpolate(project(feature), size=(h, w), mode='bilinear', align_corners=False)
                   for feature, project in zip(enhanced, self.align)]
        return sum(weights[:, i, None, None, None] * feature for i, feature in enumerate(aligned))

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels or min(x.shape[-2:]) < 1:
            raise ValueError('expected nonempty BCHW with configured channels')
        fused = self.fuse_features(x)
        return fused * (1 - self.bsu(fused))  # Eqs. 11--12, no residual bypass.
