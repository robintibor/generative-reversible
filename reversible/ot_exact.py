import numpy as np
import torch as th
import ot
from reversible.gaussian import get_gauss_samples
from reversible.util import ensure_on_same_device, np_to_var, var_to_np


def ot_emd_loss(outs, mean, std):
    gauss_samples = get_gauss_samples(len(outs), mean, std)
    diffs = outs.unsqueeze(1) - gauss_samples.unsqueeze(0)
    del gauss_samples
    diffs = th.sum(diffs * diffs, dim=2)

    transport_mat = ot.emd([],[], var_to_np(diffs))
    # sometimes weird low values, try to prevent them
    transport_mat = transport_mat * (transport_mat > (1.0/(diffs.numel())))

    transport_mat = np_to_var(transport_mat, dtype=np.float32)
    diffs, transport_mat = ensure_on_same_device(diffs, transport_mat)
    eps = 1e-6
    loss = th.sqrt(th.sum(transport_mat * diffs) + eps)
    return loss


def ot_emd_loss_for_samples(samples_a, samples_b):
    diffs = samples_a.unsqueeze(1) - samples_b.unsqueeze(0)
    diffs = th.sum(diffs * diffs, dim=2)

    transport_mat = ot.emd([], [], var_to_np(diffs))
    # sometimes weird low values, try to prevent them
    transport_mat = transport_mat * (transport_mat > (1.0/(diffs.numel())))

    transport_mat = np_to_var(transport_mat, dtype=np.float32)
    diffs, transport_mat = ensure_on_same_device(diffs, transport_mat)
    eps = 1e-6
    loss = th.sqrt(th.sum(transport_mat * diffs) + eps)
    return loss


def ot_euclidean_loss(outs, mean, std, normalize_by_global_emp_std=False):
    gauss_samples = get_gauss_samples(len(outs), mean, std)

    diffs = outs.unsqueeze(1) - gauss_samples.unsqueeze(0)
    del gauss_samples
    if normalize_by_global_emp_std:
        global_emp_std = th.mean(th.std(outs, dim=0))
        diffs = diffs / global_emp_std
    diffs = th.sqrt(th.clamp(th.sum(diffs * diffs, dim=2), min=1e-6))

    transport_mat = ot.emd([],[], var_to_np(diffs))
    # sometimes weird low values, try to prevent them
    transport_mat = transport_mat * (transport_mat > (1.0/(diffs.numel())))

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
    diffs = samples_a.unsqueeze(1) - samples_b.unsqueeze(0)
    diffs = th.sqrt(th.clamp(th.sum(diffs * diffs, dim=2), min=1e-6))

    transport_mat = ot.emd([], [], var_to_np(diffs))
    # sometimes weird low values, try to prevent them
    transport_mat = transport_mat * (transport_mat > (1.0/(diffs.numel())))

    transport_mat = np_to_var(transport_mat, dtype=np.float32)
    diffs, transport_mat = ensure_on_same_device(diffs, transport_mat)
    loss = th.sum(transport_mat * diffs)
    return loss


def ot_euclidean_transport_mat(samples_a, samples_b):
    diffs = samples_a.unsqueeze(1) - samples_b.unsqueeze(0)
    diffs = th.sqrt(th.clamp(th.sum(diffs * diffs, dim=2), min=1e-6))

    transport_mat = ot.emd([], [], var_to_np(diffs))
    # sometimes weird low values, try to prevent them
    transport_mat = transport_mat * (transport_mat > (1.0/(diffs.numel())))

    transport_mat = np_to_var(transport_mat, dtype=np.float32)
    diffs, transport_mat = ensure_on_same_device(diffs, transport_mat)
    return transport_mat


def ot_squared_diff_transport_mat(samples_a, samples_b):
    diffs = samples_a.unsqueeze(1) - samples_b.unsqueeze(0)
    diffs = th.sum(diffs * diffs, dim=2)

    transport_mat = ot.emd([], [], var_to_np(diffs))
    # sometimes weird low values, try to prevent them
    transport_mat = transport_mat * (transport_mat > (1.0/(diffs.numel())))

    transport_mat = np_to_var(transport_mat, dtype=np.float32)
    diffs, transport_mat = ensure_on_same_device(diffs, transport_mat)
    return transport_mat


def get_wanted_points(samples_to_move, samples_to_match, ot_transport_mat_fn):
    transport_mat = ot_transport_mat_fn(samples_to_move, samples_to_match)

    transport_mat = transport_mat / th.sum(transport_mat, dim=1, keepdim=True)

    wanted_points = th.sum(transport_mat.unsqueeze(2) * samples_to_match.unsqueeze(0), dim=1)
    return wanted_points


def get_wanted_points_from_transport_mat(transport_mat, samples_matched):
    transport_mat = transport_mat / th.sum(transport_mat, dim=1, keepdim=True)

    wanted_points = th.sum(
        transport_mat.unsqueeze(2) * samples_matched.unsqueeze(0), dim=1)
    return wanted_points


def transport_mat_from_diffs(diffs):
    transport_mat = ot.emd([], [], var_to_np(diffs))
    # sometimes weird low values, try to prevent them
    transport_mat = transport_mat * (transport_mat > (1.0/(diffs.numel())))

    transport_mat = np_to_var(transport_mat, dtype=np.float32)
    diffs, transport_mat = ensure_on_same_device(diffs, transport_mat)
    return transport_mat