import torch
import torch.nn as nn
import numpy as np
from scipy.special import erfinv


class ProxyNorm2d(nn.Module):
    def __init__(self, in_channels, eps=0.03, act=nn.ReLU(), bias=False):
        super(ProxyNorm2d, self).__init__()

        self.in_channels = in_channels

        self.beta = nn.Parameter(torch.zeros(1, in_channels, 1, 1, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.ones(1, in_channels, 1, 1, dtype=torch.float32))

        self.eps = eps
        self.act = act
        self.bias = bias

    def uniformly_sample(self, n: int):
        assert n > 0, "n should be greater than 0"
        return np.sqrt(2) * erfinv(2 * (np.arange(n) + self.eps) / float(n) - 1)

    def fold(self, x):
        return x.flatten(2).transpose(2, 1)

    def forward(self, x):
        assert len(x.shape) in [3, 4], "Input tensor should be 3D or 4D"

        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        z = self.gamma * x + self.beta
        z = self.act(z)

        proxy_y = torch.tensor(
            self.uniformly_sample(self.in_channels),
            dtype=torch.float32,
            device=x.device,
        ).view(self.in_channels, 1, 1, 1)

        proxy_z = self.act(self.gamma * proxy_y + self.beta)

        mean = torch.mean(proxy_z, dim=0, keepdim=True, dtype=torch.float32)
        var = torch.var(proxy_z, dim=0, keepdim=True, unbiased=not self.bias)
        std = torch.rsqrt(var + self.eps)

        tilde_z = (z - mean) * std

        return tilde_z

class ProxyNorm1d(nn.Module):
    def __init__(self, in_features, eps=0.03, act=nn.ReLU(), bias=False):
        super(ProxyNorm1d, self).__init__()

        self.in_features = in_features

        self.beta = nn.Parameter(torch.zeros(1, in_features, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.ones(1, in_features, dtype=torch.float32))

        self.eps = eps
        self.act = act
        self.bias = bias

    def uniformly_sample(self, n: int):
        assert n > 0, "n should be greater than 0"
        return np.sqrt(2) * erfinv(2 * (np.arange(n) + self.eps) / float(n) - 1)

    def forward(self, x):
        assert len(x.shape) in [2, 3], "Input tensor should be 2D or 3D"

        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        z = self.gamma * x + self.beta
        z = self.act(z)

        proxy_y = torch.tensor(
            self.uniformly_sample(self.in_features),
            dtype=torch.float32,
            device=x.device,
        ).view(self.in_features, 1)

        proxy_z = self.act(self.gamma * proxy_y + self.beta)

        mean = torch.mean(proxy_z, dim=0, keepdim=True, dtype=torch.float32)
        var = torch.var(proxy_z, dim=0, keepdim=True, unbiased=not self.bias)
        std = torch.rsqrt(var + self.eps)

        tilde_z = (z - mean) * std

        return tilde_z