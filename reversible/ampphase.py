import torch as th
import numpy as np

from reversible.gaussian import standard_gaussian_icdf, standard_gaussian_cdf


class AmplitudePhase(th.nn.Module):
    def __init__(self):
        super(AmplitudePhase, self).__init__()

    def forward(self, x):
        n_chans = x.size()[1]
        assert n_chans % 2 == 0
        x1 = x[:, :n_chans // 2]
        x2 = x[:, n_chans // 2:]
        amps, phases = to_amp_phase(x1, x2)
        return th.cat((amps, phases), dim=1)


def to_amp_phase(x, y):
    amps = th.sqrt((x * x) + (y * y))
    phases = th.atan2(y, x)
    return amps, phases


def amp_phase_to_x_y(amps, phases):
    x, y = th.cos(phases), th.sin(phases)

    x = x * amps
    y = y * amps
    return x, y


def compute_mean_phase(phases, this_weights):
    x, y = th.cos(phases), th.sin(phases)
    this_probs = this_weights / th.sum(this_weights)
    mean_x = th.sum(this_probs.unsqueeze(1) * x, dim=0)
    mean_y = th.sum(this_probs.unsqueeze(1) * y, dim=0)
    mean_phase = th.atan2(mean_y, mean_x)
    return mean_phase


def set_phase_interval_around_mean(phases, mean_phase):
    mean_phase = mean_phase.unsqueeze(0)
    return ((phases - mean_phase + np.pi) % (2 * np.pi)) - np.pi + mean_phase


def set_phase_interval_around_mean_in_outs(outs, this_weights=None, means=None,):
    assert (this_weights is None) != (means is None)
    n_chans = outs.size()[1]
    assert n_chans % 2 == 0
    amps = outs[:, :n_chans // 2]
    phases = outs[:, n_chans // 2:]
    if this_weights is not None:
        mean_phase = compute_mean_phase(phases, this_weights)
    else:
        assert means.size()[0] == n_chans
        mean_phase = means[n_chans // 2:]
    demeaned_phases = set_phase_interval_around_mean(phases, mean_phase)
    new_outs = th.cat((amps, demeaned_phases), dim=1)
    return new_outs

def uniform_to_gaussian_phases_in_outs(outs_with_demeaned_phases, means):
    n_chans = outs_with_demeaned_phases.size()[1]
    assert n_chans % 2 == 0
    amps = outs_with_demeaned_phases[:, :n_chans // 2]
    phases = outs_with_demeaned_phases[:, n_chans // 2:]
    mean_phases = means[n_chans // 2:]
    gauss_phases = uniform_to_gaussian_phases(phases, mean_phases)
    outs = th.cat((amps, gauss_phases), dim=1)
    return outs


def uniform_to_gaussian_phases(phases, mean_phases):
    eps = 1e-6#1e-7#1e-7 # 1e-8 results in -inf for icdf of gaussian.....
    start = mean_phases - np.pi - eps
    stop = mean_phases + np.pi + eps
    cdfs = (phases - start.unsqueeze(0)) / (stop - start).unsqueeze(0)
    icdfs = standard_gaussian_icdf(cdfs)
    return (icdfs * (stop - start).unsqueeze(0)) + mean_phases.unsqueeze(0)


def gaussian_to_uniform_phases_in_outs(outs_with_gaussian_phases, means):
    n_chans = outs_with_gaussian_phases.size()[1]
    assert n_chans % 2 == 0
    amps = outs_with_gaussian_phases[:, :n_chans // 2]
    phases = outs_with_gaussian_phases[:, n_chans // 2:]
    mean_phases = means[n_chans // 2:]
    uni_phases = gaussian_to_uniform_phases(phases, mean_phases)
    outs = th.cat((amps, uni_phases), dim=1)
    return outs


def gaussian_to_uniform_phases(phases, mean_phases):
    eps = 1e-6#1e-7#1e-7 # 1e-8 results in -inf for icdf of gaussian.....
    start = mean_phases - np.pi - eps
    stop = mean_phases + np.pi + eps
    icdfs = (phases - mean_phases.unsqueeze(0)) / (stop - start).unsqueeze(0)
    cdfs = standard_gaussian_cdf(icdfs)
    uni_phases = (cdfs * (stop - start).unsqueeze(0)) + start.unsqueeze(0)
    return uni_phases
