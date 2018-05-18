import ot
import numpy as np
import torch as th
from reversible.util import np_to_var, var_to_np, ensure_on_same_device


def unbalanced_transport_mat_squared_diff(samples_a, samples_b, cover_fraction,
                                          return_diffs=False):
    diffs = samples_a.unsqueeze(1) - samples_b.unsqueeze(0)
    diffs = th.sum(diffs * diffs, dim=2)
    # add dummy point with distance 0 to everything
    dummy =  th.zeros_like(diffs[0:1,:])
    diffs = th.cat((diffs, dummy), dim=0)
    a = np.ones(len(samples_a)) / len(samples_a) * cover_fraction
    a = np.concatenate((a, [1 - cover_fraction]))
    transport_mat = ot.emd(a, [], var_to_np(diffs))
    transport_mat = np_to_var(transport_mat, dtype=np.float32)
    transport_mat, diffs = ensure_on_same_device(transport_mat, diffs)
    if return_diffs:
        return transport_mat, diffs
    else:
        return transport_mat


def get_unbalanced_squared_diffs(samples_a, samples_b, cover_fraction):
    diffs = samples_a.unsqueeze(1) - samples_b.unsqueeze(0)
    diffs = th.sum(diffs * diffs, dim=2)
    # add dummy point with distance 0 to everything
    dummy =  th.autograd.Variable(th.zeros_like(diffs[0:1,:]))
    diffs = th.cat((diffs, dummy), dim=0)
    a = np.ones(len(samples_a)) / len(samples_a) * cover_fraction
    a = np.concatenate((a, [1 - cover_fraction]))
    return diffs, a


def only_used_tmat_diffs(t_mat, diffs):
    mask = t_mat[-1] > 0
    used_tmat = t_mat[(mask^1).unsqueeze(0)].view(t_mat.size()[0], -1)

    used_diffs = diffs[(mask^1).unsqueeze(0)].view(diffs.size()[0], -1)
    return used_tmat, used_diffs, mask
