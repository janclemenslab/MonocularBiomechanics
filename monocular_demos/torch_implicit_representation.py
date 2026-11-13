"""PyTorch trajectory representations with the original encoding and layout."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def calculate_encoding_length(max_time, min_freq=80):
    return max(0, math.ceil(math.log2(min_freq * float(max_time) / (2 * math.pi)) + 1))


def positional_encoding(inputs, positional_encoding_dims=3):
    frequencies = 2.0 ** torch.arange(positional_encoding_dims, device=inputs.device, dtype=inputs.dtype)
    angles = frequencies[:, None, None] * inputs.reshape(-1, 1)[None]
    periodic = torch.stack((angles.sin(), angles.cos())).transpose(0, 2).reshape(inputs.shape[0], -1)
    return torch.cat((inputs, periodic), -1)


class Linear(nn.Linear):
    def __init__(self, in_features, out_features, *, compute_dtype=torch.float16, custom_bias=None):
        super().__init__(in_features, out_features)
        self.compute_dtype = compute_dtype
        if custom_bias is not None:
            bias = torch.as_tensor(custom_bias, dtype=self.bias.dtype)
            with torch.no_grad():
                self.bias.copy_(torch.where(bias.isnan(), self.bias, bias))

    def forward(self, x):
        return x.to(self.compute_dtype) @ self.weight.to(self.compute_dtype).T + self.bias.to(self.compute_dtype)


class ImplicitTrajectory(nn.Module):
    def __init__(self, features=(128, 256, 512, 1024, 2048, 2048, 4096), joints=75,
                 spatial_dims=3, concatenate_layers=3, encoding_length=8, max_time=60,
                 layer_norm=False, custom_bias=None,
                 jmp_policy='params=float32,compute=float32,output=float32', non_linearity='relu', *, key=0):
        super().__init__()
        if not features or not 0 <= concatenate_layers < len(features):
            raise ValueError('concatenate_layers must be smaller than the number of hidden layers.')
        if max_time <= 0:
            raise ValueError('max_time must be positive.')
        policy = dict(part.split('=') for part in str(jmp_policy).split(','))
        self.compute_dtype = getattr(torch, policy['compute'])
        self.output_dtype = getattr(torch, policy['output'])
        self.joints, self.spatial_dims = joints, spatial_dims
        self.concatenate_layers, self.encoding_length = concatenate_layers, encoding_length
        self.layer_norm = layer_norm
        self.non_linearity = non_linearity
        self.max_time = nn.Parameter(torch.tensor(math.ceil(max_time), dtype=torch.float32))
        encoding_size = encoding_length * 2 + 1
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(key))
            self.layers = nn.ModuleList()
            for i, width in enumerate(features):
                input_dim = encoding_size if i == 0 else features[i - 1]
                if 0 < i <= concatenate_layers:
                    input_dim += encoding_size
                self.layers.append(Linear(input_dim, width, compute_dtype=self.compute_dtype))
            self.final = Linear(features[-1], joints * spatial_dims, compute_dtype=self.compute_dtype, custom_bias=custom_bias)
        self.to(dtype=getattr(torch, policy['params']))

    def forward(self, time_point):
        shape = time_point.shape
        encoded = positional_encoding((time_point.to(self.max_time) / self.max_time * torch.pi).reshape(-1, 1), self.encoding_length)
        encoded = encoded.to(self.compute_dtype)
        x = encoded
        activation = {'sin': torch.sin, 'tanh': torch.tanh, 'sigmoid': torch.sigmoid,
                      'swish': F.silu}.get(self.non_linearity)
        if activation is None:
            activation = getattr(F, self.non_linearity) if isinstance(self.non_linearity, str) else self.non_linearity
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.layer_norm:
                x = F.layer_norm(x, (x.shape[-1],), eps=1e-5)
            x = activation(x)
            if i < self.concatenate_layers:
                x = torch.cat((x, encoded), -1)
        return self.final(x).reshape(*shape, self.joints, self.spatial_dims).to(self.output_dtype)


class CubicSplineTrajectory(nn.Module):
    def __init__(self, sequence_length, joints=75, spatial_dims=3, max_time=60, *, key=0):
        super().__init__()
        if sequence_length < 2 or max_time <= 0:
            raise ValueError('Cubic trajectories require at least two points and positive max_time.')
        self.sequence_length, self.joints, self.spatial_dims = sequence_length, joints, spatial_dims
        self.max_time = float(max_time)
        generator = torch.Generator().manual_seed(int(key))
        self.control_points = nn.Parameter(torch.randn(sequence_length, joints, spatial_dims, generator=generator) * 0.01)

    def forward(self, time_point):
        scaled = (time_point / self.max_time).clamp(0, 1) * (self.sequence_length - 1)
        index = scaled.floor().long()
        s = (scaled - index)[..., None, None]
        p0, p1, p2, p3 = (self.control_points[(index + offset).clamp(0, self.sequence_length - 1)] for offset in (-1, 0, 1, 2))
        return 0.5 * (2 * p1 + (-p0 + p2) * s + (2 * p0 - 5 * p1 + 4 * p2 - p3) * s**2 + (-p0 + 3 * p1 - 3 * p2 + p3) * s**3)
