import numpy as np
import torch as th
import ot
from reversible.gaussian import get_gauss_samples
from reversible.util import ensure_on_same_device, np_to_var, var_to_np


def ot_emd_loss(outs, mean, std):
    gauss_samples = get_gauss_samples(len(outs), mean, std)


    diffs = outs.unsqueeze(0) - gauss_samples.unsqueeze(1)
    del gauss_samples
    diffs = th.sum(diffs * diffs, dim=2)

    transport_mat = ot.emd([],[], var_to_np(diffs))

    transport_mat = np_to_var(transport_mat, dtype=np.float32)
    diffs, transport_mat = ensure_on_same_device(diffs, transport_mat)
    loss = th.sqrt(th.sum(transport_mat * diffs))
    return loss


def ot_emd_loss_for_samples(samples_a, samples_b):
    diffs = samples_a.unsqueeze(0) - samples_b.unsqueeze(1)
    diffs = th.sum(diffs * diffs, dim=2)

    transport_mat = ot.emd([], [], var_to_np(diffs))

    transport_mat = np_to_var(transport_mat, dtype=np.float32)
    diffs, transport_mat = ensure_on_same_device(diffs, transport_mat)
    loss = th.sqrt(th.sum(transport_mat * diffs))
    return loss


def ot_euclidean_loss(outs, mean, std, normalize_by_global_emp_std=False):
    gauss_samples = get_gauss_samples(len(outs), mean, std)

    diffs = outs.unsqueeze(0) - gauss_samples.unsqueeze(1)
    del gauss_samples
    if normalize_by_global_emp_std:
        global_emp_std = th.mean(th.std(outs, dim=0))
        diffs = diffs / global_emp_std
    diffs = th.sqrt(th.clamp(th.sum(diffs * diffs, dim=2), min=1e-6))

    transport_mat = ot.emd([],[], var_to_np(diffs))

    transport_mat = np_to_var(transport_mat, dtype=np.float32)
    diffs, transport_mat = ensure_on_same_device(diffs, transport_mat)
    loss = th.sum(transport_mat * diffs)
    return loss


def ot_euclidean_energy_loss(outs, mean, std):
    gauss_samples = get_gauss_samples(len(outs), mean, std)
    o1, o2 = th.chunk(outs, 2, dim=0)
    g1, g2 = th.chunk(gauss_samples, 2, dim=0)
    return ot_eucledian_energy_loss(o1,o2, g1,g2)


def ot_eucledian_energy_loss(a1,a2,b1,b2):
    loss = 2 * ot_euclidean_loss_for_samples(a1,b1) -  (
        ot_euclidean_loss_for_samples(a1,a2) -
        ot_euclidean_loss_for_samples(b1,b2))
    return loss


def ot_euclidean_loss_for_samples(samples_a, samples_b):
    diffs = samples_a.unsqueeze(0) - samples_b.unsqueeze(1)
    diffs = th.sqrt(th.clamp(th.sum(diffs * diffs, dim=2), min=1e-6))

    transport_mat = ot.emd([], [], var_to_np(diffs))

    transport_mat = np_to_var(transport_mat, dtype=np.float32)
    diffs, transport_mat = ensure_on_same_device(diffs, transport_mat)
    loss = th.sum(transport_mat * diffs)
    return loss
