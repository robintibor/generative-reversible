import ot
import numpy as np
import torch as th
from reversible.util import np_to_var, var_to_np


def unbalanced_transport_mat_squared_diff(samples_a, samples_b, cover_fraction):
    diffs = samples_a.unsqueeze(1) - samples_b.unsqueeze(0)
    diffs = th.sum(diffs * diffs, dim=2)
    # add dummy point with distance 0 to everything
    diffs = th.cat((diffs, th.autograd.Variable(th.zeros(1,diffs.size()[1]))), dim=0)
    a = np.ones(len(samples_a)) / len(samples_a) * cover_fraction
    a = np.concatenate((a, [1 - cover_fraction]))
    transport_mat = ot.emd(a, [], var_to_np(diffs))
    transport_mat = np_to_var(transport_mat, dtype=np.float32)
    return transport_mat

