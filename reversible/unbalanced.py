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
    mask = t_mat[-1] > 1e-10
    used_tmat = t_mat[(mask^1).unsqueeze(0)].view(t_mat.size()[0], -1)

    used_diffs = diffs[(mask^1).unsqueeze(0)].view(diffs.size()[0], -1)
    return used_tmat, used_diffs, mask


def unbalanced_transport(to_move, to_match, cover_fraction):
    all_diffs = to_move.unsqueeze(1) - to_match.unsqueeze(0)

    all_diffs = th.sum(all_diffs * all_diffs, dim=2)
    sorted_diffs, i_sorted = th.sort(all_diffs, dim=1)
    del all_diffs

    n_to_cover = int(np.round(cover_fraction * to_match.size()[0]))
    n_max_cut_off = n_to_cover
    n_step = n_max_cut_off // 20
    cut_off_found = False
    uniques_so_far = []
    for i_cut_off in range(0, n_max_cut_off, n_step):
        i_sorted_part = i_sorted[:,
                        i_cut_off:i_cut_off + n_step].contiguous().view(-1)
        this_unique_inds = np.unique(var_to_np(i_sorted_part))
        uniques_so_far = np.unique(
            np.concatenate((uniques_so_far, this_unique_inds)))
        if len(uniques_so_far) > n_to_cover:
            i_cut_off = i_cut_off + n_step
            i_cut_off = i_cut_off * 2
            cut_off_found = True
            break

    if not cut_off_found:
        i_cut_off = n_max_cut_off
    i_cut_off = np.minimum(i_cut_off, n_max_cut_off)

    i_sorted_part = i_sorted[:, :i_cut_off].contiguous().view(-1)

    unique_inds = np.unique(var_to_np(i_sorted_part))
    unique_inds = np_to_var(unique_inds, dtype=np.int64).cuda()
    part_to_match = to_match[unique_inds]

    part_cover_fraction = float(n_to_cover / float(part_to_match.size()[0]))
    assert cover_fraction > 0 and cover_fraction <= 1

    t_mat, diffs = unbalanced_transport_mat_squared_diff(
        to_move, part_to_match,
        cover_fraction=part_cover_fraction,
        return_diffs=True)

    t_mat, diffs, mask = only_used_tmat_diffs(t_mat, diffs)
    used_sample_inds = unique_inds[mask ^ 1]
    t_mat = t_mat[:-1]
    diffs = diffs[:-1]
    t_mat = t_mat / th.sum(t_mat)
    loss = th.sum(t_mat * diffs)
    rejected_mask = th.ones_like(to_match[:, 0] > 0)

    rejected_mask[used_sample_inds] = 0
    return loss, rejected_mask
