import torch as th
import numpy as np

from reversible.schedulers import ScheduledOptimizer, DivideSqrtUpdates
from reversible.util import var_to_np, np_to_var, ensure_on_same_device


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


def get_i_example_to_i_samples(diffs, v, optimizer_v, stop_min_bincount,
                               stop_max_bincount, max_iters,
                               collect_out_samples):
    diffs = diffs.detach()
    if max_iters == 0:
        # just for collection
        min_diffs, inds = th.min(diffs - v.unsqueeze(1), dim=0)

    for i_iter in range(max_iters):
        min_diffs, inds = th.min(diffs - v.unsqueeze(1), dim=0)
        bincounts = np.bincount(var_to_np(inds), minlength=len(v))
        if np.min(bincounts) >= stop_min_bincount:
            break
        if np.max(bincounts) <= stop_max_bincount:
            break
        loss = -(th.mean(min_diffs) + th.mean(v))
        optimizer_v.zero_grad()
        loss.backward()
        optimizer_v.step()
        v.data = v.data - th.mean(v.data)
    if collect_out_samples:
        i_example_to_i_samples = [[] for _ in range(diffs.size()[0])]
        i_example_to_diffs = [[] for _ in range(diffs.size()[0])]
        for i_sample, i_out in enumerate(inds):
            i_example_to_i_samples[var_to_np(i_out)[0]].append(i_sample)
            i_example_to_diffs[var_to_np(i_out)[0]].append(min_diffs[i_sample])
    else:
        i_example_to_i_samples = None
        i_example_to_diffs = None
    return i_example_to_i_samples, i_example_to_diffs


def match_by_semidual_adam(diffs,v, stop_min_bincount=1,stop_max_bincount=4,
        max_iters=200, collect_out_samples=True):
    init_lr = var_to_np(th.mean(diffs) * 0.02)[0]
    optimizer_v = th.optim.Adam([v], lr=init_lr)

    i_example_to_i_samples, i_example_to_diffs = get_i_example_to_i_samples(
        diffs,v, optimizer_v, stop_min_bincount=stop_min_bincount,
        stop_max_bincount=stop_max_bincount,
        collect_out_samples=collect_out_samples, max_iters=max_iters)
    return i_example_to_i_samples


def restrict_samples_to_max_n_per_example(i_example_to_i_samples, i_example_to_i_samples_2, max_n):
    # reduce both arrays correctly
    for i_example in range(len(i_example_to_i_samples)):
        remaining_n = max_n
        i_example_to_i_samples[i_example] = i_example_to_i_samples[i_example][:remaining_n]
        remaining_n -= len(i_example_to_i_samples[i_example])
        i_example_to_i_samples_2[i_example] = i_example_to_i_samples_2[i_example][:remaining_n]
    return i_example_to_i_samples, i_example_to_i_samples_2


def merge_samples(i_example_to_i_samples, i_example_to_i_samples_2, gauss_samples,
                 gauss_samples_2):
    inds_a = np.sort(np.concatenate(i_example_to_i_samples))
    th_inds_a = np_to_var(inds_a, dtype=np.int64)
    th_inds_a,  _ = ensure_on_same_device(
        th_inds_a, gauss_samples)
    samples_a = gauss_samples[th_inds_a]
    inds_b = np.sort(np.concatenate(i_example_to_i_samples_2))
    if len(inds_b) > 0:
        th_inds_b = np_to_var(inds_b, dtype=np.int64)
        th_inds_b,  _ = ensure_on_same_device(
            th_inds_b, gauss_samples)
        samples_b = gauss_samples_2[th_inds_b]
    a_dict = dict([(val, i) for i,val in enumerate(inds_a)])
    b_dict = dict([(val, i + len(a_dict)) for i,val in enumerate(inds_b)])
    # merge samples
    i_example_to_i_samples_merged = []
    for i_example in range(len(i_example_to_i_samples)):
        a_examples = [a_dict[i] for i in i_example_to_i_samples[i_example]]
        b_examples = [b_dict[i] for i in i_example_to_i_samples_2[i_example]]
        i_example_to_i_samples_merged.append(a_examples + b_examples)
    if len(inds_b) > 0:
        all_samples = th.cat((samples_a, samples_b), dim=0)
    else:
        all_samples = samples_a
    return all_samples, i_example_to_i_samples_merged


def collect_out_to_samples(diffs, v):
    min_diffs, inds = th.min(diffs - v.unsqueeze(1), dim=0)
    inds = var_to_np(inds)
    i_example_to_i_samples = [[] for _ in range(diffs.size()[0])]
    i_example_to_diffs = [[] for _ in range(diffs.size()[0])]
    for i_sample, i_out in enumerate(inds):
        i_example_to_i_samples[i_out].append(i_sample)
        i_example_to_diffs[i_out].append(min_diffs[i_sample])
    return i_example_to_i_samples, i_example_to_diffs


def collect_samples(outs, v, sample_fn, n_max, iters):
    samples = sample_fn()
    diffs = th.sum(
        (outs.unsqueeze(dim=1) - samples.unsqueeze(dim=0)) ** 2,
        dim=2)
    i_example_to_i_samples, _ = collect_out_to_samples(diffs, v)
    bincounts = []
    bincounts.append(np.array([len(a) for a in i_example_to_i_samples]))
    for _ in range(iters):
        samples_2 = sample_fn()
        diffs_2 = th.sum((outs.unsqueeze(dim=1) - samples_2.unsqueeze(
                             dim=0)) ** 2, dim=2)
        i_example_to_i_samples_2, _ = collect_out_to_samples(diffs_2, v)
        bincounts.append(np.array([len(a) for a in i_example_to_i_samples_2]))
        i_example_to_i_samples, i_example_to_i_samples_2 = restrict_samples_to_max_n_per_example(
            i_example_to_i_samples, i_example_to_i_samples_2, n_max)
        samples, i_example_to_i_samples = merge_samples(
            i_example_to_i_samples, i_example_to_i_samples_2,
            samples, samples_2)
        lens = np.array([len(a) for a in i_example_to_i_samples])
        assert len(samples) == np.sum(lens), (
            "{:d} samples but {:d} assignments".format(len(samples), np.sum(lens)))
        if len(samples) == n_max * len(outs):
            assert np.all(lens == n_max), (
                "Unique Lengths: {:s} from {:d} samples of {:d} samples".format(str(np.unique(lens)),
                len(i_example_to_i_samples), len(samples)))
            break
    return samples, bincounts


def sample_match_and_bincount(outs, v, sample_fn, iters):
    samples = sample_fn()
    diffs = th.sum(
        (outs.unsqueeze(dim=1) - samples.unsqueeze(dim=0)) ** 2,
        dim=2)
    i_example_to_i_samples, _ = collect_out_to_samples(diffs, v)
    bincounts = []
    bincounts.append(np.array([len(a) for a in i_example_to_i_samples]))
    for _ in range(iters):
        samples_2 = sample_fn()
        diffs_2 = th.sum((outs.unsqueeze(dim=1) - samples_2.unsqueeze(
            dim=0)) ** 2, dim=2)
        i_example_to_i_samples_2, _ = collect_out_to_samples(diffs_2, v)
        bincounts.append(np.array([len(a) for a in i_example_to_i_samples_2]))
    return bincounts

def optimize_v_optimizer(v, optim_v, outs, sample_fn, max_iters=150):
    exp_bin_counts = None
    avg_v = th.zeros_like(v.data)
    for i_update in range(max_iters):
        samples = sample_fn()
        diffs = th.sum((outs.unsqueeze(dim=1) - samples.unsqueeze(dim=0)) ** 2, dim=2)
        min_diffs, inds = th.min(diffs - v.unsqueeze(1), dim=0)
        loss = -(th.mean(min_diffs) + th.mean(v))
        optim_v.zero_grad()
        loss.backward()
        optim_v.step()
        v.data = v.data - th.mean(v.data)
        k = i_update + 1
        avg_v = ((k-1) / k) * avg_v + (1/k) * v.data
    return i_update,  avg_v


def optimize_v_adaptively(outs, v, sample_fn, bin_dev_threshold):
    # Optimize V
    n_updates_total = 0
    outs = outs.detach()
    gauss_samples = sample_fn()
    diffs = th.sum((outs.unsqueeze(dim=1) - gauss_samples.unsqueeze(dim=0)) ** 2, dim=2)
    init_lr = float(var_to_np(th.mean(th.min(diffs, dim=1)[0]))[0])
    optim_v_orig = th.optim.SGD([v], lr=init_lr)
    optim_v = ScheduledOptimizer(DivideSqrtUpdates(), optim_v_orig, True)

    # for now:
    # do 20, with lr= mean(mindiffs)
    # after, do 20, check l1 with 8, do 20, check l1 with 8
    # if below 1, finish

    i_updates, avg_v = optimize_v_optimizer(
      v, optim_v, outs.detach(), sample_fn, max_iters=25)
    v.data = avg_v
    n_updates_total += i_updates + 1
    for _ in range(10):
        i_updates, avg_v = optimize_v_optimizer(
            v, optim_v, outs.detach(), sample_fn, max_iters=20)
        n_updates_total += i_updates + 1
        v.data = avg_v
        bincounts = sample_match_and_bincount(outs.detach(), v, sample_fn, iters=10)
        bin_dev = np.mean(np.abs(bincounts - np.mean(bincounts)))
        if bin_dev < bin_dev_threshold:
            break
    return bincounts, bin_dev, n_updates_total


