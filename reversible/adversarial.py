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


def take_only_large_stds(l_out, std, n_wanted_stds):
    i_stds = th.sort(std)[1][-n_wanted_stds:]
    l_out = l_out.index_select(index=i_stds, dim=1)
    return l_out


def possibly_reduce_to_wanted_stds(l_out, layer_adv, std, n_wanted_stds):
    if layer_adv[0].module.in_features != l_out.size()[1]:
        assert layer_adv[0].module.in_features == n_wanted_stds
        l_out = take_only_large_stds(l_out, std, n_wanted_stds)
    return l_out


def d_losses_layer(layer_advs, l_outs_real, l_outs_fake):
    if len(layer_advs) > 0:
        d_losses = []
        for layer_adv, l_out_real, l_out_fake in list(
                zip(layer_advs, l_outs_real, l_outs_fake)):
            layer_adv.train()
            l_out_real = possibly_reduce_to_wanted_stds(
                l_out_real, layer_adv, std, n_wanted_stds)
            l_out_fake = possibly_reduce_to_wanted_stds(
                l_out_fake, layer_adv, std, n_wanted_stds)
            assert layer_adv[0].module.in_features == l_out_real.size()[1] == \
                   l_out_fake.size()[1]
            score_real = layer_adv(
                choose(pixels_to_batch(l_out_real), n_max=5000))
            score_fake = layer_adv(
                choose(pixels_to_batch(l_out_fake), n_max=5000))
            d_loss = nn.functional.relu(
                1.0 - score_real).mean() + nn.functional.relu(
                1.0 + score_fake).mean()
            d_losses.append(d_loss)
        d_loss = th.mean(th.cat(d_losses))
    else:
        d_loss = 0
        d_losses = []
    return d_loss, d_losses


def g_losses_layer(layer_advs, l_outs_real, l_outs_fake):
    if len(layer_advs) > 0:
        g_losses_real = []
        for layer_adv, l_out_real in list(zip(layer_advs, l_outs_real)):
            layer_adv.eval()
            l_out_real = possibly_reduce_to_wanted_stds(
                l_out_real, layer_adv, std, n_wanted_stds)
            score_real = layer_adv(
                choose(pixels_to_batch(l_out_real), n_max=5000))
            g_loss = th.mean(score_real)
            g_losses_real.append(g_loss)
        g_loss_real = th.mean(th.cat(g_losses_real))

        g_losses_fake = []
        for layer_adv, l_out_fake in list(zip(layer_advs, l_outs_fake)):
            layer_adv.eval()
            l_out_fake = possibly_reduce_to_wanted_stds(
                l_out_fake, layer_adv, std, n_wanted_stds)
            score_fake = layer_adv(
                choose(pixels_to_batch(l_out_fake), n_max=5000))
            g_loss = -th.mean(score_fake)
            g_losses_fake.append(g_loss)
        g_loss_fake = th.mean(th.cat(g_losses_fake))
    else:
        g_loss_real = 0
        g_loss_fake = 0
        g_losses_real = []
        g_losses_fake = []
    return g_loss_real, g_loss_fake, g_losses_real, g_losses_fake
