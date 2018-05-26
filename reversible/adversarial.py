from torch import nn
import torch as th

from reversible.spectral_norm import SpectralNorm


def create_adv(dim, intermediate_dim, snorm):
    if snorm is None:
        model = nn.Sequential(
            nn.Linear(dim, intermediate_dim),
            ConcatReLU(),
            nn.Linear(intermediate_dim*2, 1))
    else:
        model = nn.Sequential(
            SpectralNorm(nn.Linear(dim, intermediate_dim), power_iterations=1, to_norm=snorm),
            ConcatReLU(),
            SpectralNorm(nn.Linear(intermediate_dim*2, 1), power_iterations=1, to_norm=snorm))

    model = model.cuda()
    return model


def create_adv_2_layer(dim, intermediate_dim, snorm):
    if snorm is None:
        model = nn.Sequential(
            nn.Linear(dim, intermediate_dim),
            ConcatReLU(),
            nn.Linear(intermediate_dim*2, intermediate_dim*2),
            nn.ReLU(),
            nn.Linear(intermediate_dim*2, 1))
    else:
        model = nn.Sequential(
            SpectralNorm(nn.Linear(dim, intermediate_dim), power_iterations=1, to_norm=snorm),
            ConcatReLU(),
            SpectralNorm(nn.Linear(intermediate_dim*2, intermediate_dim*2), power_iterations=1, to_norm=snorm),
            nn.ReLU(),
            SpectralNorm(nn.Linear(intermediate_dim*2, 1), power_iterations=1, to_norm=snorm))

    model = model.cuda()
    return model


class ConcatReLU(nn.Module):
    def __init__(self):
        super(ConcatReLU, self).__init__()

    def forward(self, x):
        return th.cat((nn.functional.relu(x), -nn.functional.relu(-x)), dim=1)
