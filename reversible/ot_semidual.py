import torch as th
import numpy as np
from reversible.util import var_to_np

def semi_dual_transport_loss(outs, mean, std):
    gauss_samples = get_gauss_samples(len(outs), mean, std)
    return semi_dual_transport_loss_for_samples(outs, gauss_samples, v_network, u_network)


def semi_dual_transport_loss_for_samples(samples_a, samples_b, a_network, b_network):
    diffs = samples_a.unsqueeze(1) - samples_b.unsqueeze(0)
    diffs = th.sum((diffs * diffs), dim=2)
    samples_a = samples_a.detach()
    samples_b = samples_b.detach()
    cur_v = a_network(samples_a).squeeze(1)
    cur_u = b_network(samples_b).squeeze(1)
    diffs_v = diffs - cur_v.unsqueeze(1)
    diffs_u = diffs - cur_u.unsqueeze(0)
    min_diffs_v, _ = th.min(diffs_v, dim=0)
    min_diffs_u, _ = th.min(diffs_u, dim=1)
    ot_dist = th.mean(cur_v) + th.mean(cur_u)  + th.mean(min_diffs_u) + th.mean(min_diffs_v)
    return ot_dist


def get_wanted_points_semi_dual(samples_discrete, samples_continuous, max_iters,
                                    min_iters,
                                min_grad_change, max_bin_count, v=None):
    all_diffs = []
    for sample_continuous in th.chunk(samples_continuous, 10, dim=0):
        diffs = (samples_discrete.unsqueeze(1) - sample_continuous.unsqueeze(0)).detach()
        diffs = th.sum(diffs ** 2, dim=2)
        all_diffs.append(diffs)
    diffs = th.cat(all_diffs, dim=1).detach()
    if v is None:
        v = th.autograd.Variable(th.zeros(diffs.size()[0]).cuda(), requires_grad=True)
    sgd = th.optim.SGD([v], lr=0.1)
    for i_iter in range(max_iters):
        min_diffs, inds = th.min(diffs - v.unsqueeze(1), dim=0)
        loss = -(th.mean(min_diffs) + th.mean(v))
        sgd.zero_grad()
        loss.backward()
        sgd.step()
        v.data = v.data - th.mean(v.data)
        if (var_to_np(th.sum(th.abs(v.grad)))[0] < min_grad_change) and (
            i_iter > min_iters):
            break
        if np.min(np.bincount(var_to_np(inds))) > max_bin_count and (
            i_iter > min_iters):
            break
    wanted_points = (samples_discrete * 0).clone()
    n_points = th.zeros(len(wanted_points)).cuda()
    for i_sample, i_out in enumerate(inds):
        wanted_points.data[i_out.data] += samples_continuous.data[i_sample]
        n_points[i_out.data] += 1

    wanted_points.data = wanted_points.data / th.clamp(n_points.unsqueeze(1),
                                                       min=1)
    if np.any(n_points.cpu().numpy() < 1):
        min_diffs, inds = th.min(diffs, dim=1)
        min_matched = samples_continuous.index_select(index=inds, dim=0)
        mask = (n_points < 1).type_as(wanted_points.data).unsqueeze(1)
        wanted_points.data = wanted_points.data + (mask * min_matched.data)

    return wanted_points, v, n_points, i_iter