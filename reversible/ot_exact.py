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