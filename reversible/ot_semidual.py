import torch as th
import numpy as np

from reversible.gaussian import get_gauss_samples
from reversible.schedulers import ScheduledOptimizer, DivideSqrtUpdates
from reversible.util import var_to_np, np_to_var, ensure_on_same_device

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
    bincounts = []
    for _ in range(iters):
        samples = sample_fn()
        diffs = th.sum((outs.unsqueeze(dim=1) - samples.unsqueeze(
            dim=0)) ** 2, dim=2)
        i_example_to_i_samples, _ = collect_out_to_samples(diffs, v)
        bincounts.append(np.array([len(a) for a in i_example_to_i_samples]))
    return bincounts


def optimize_v_optimizer(v, optim_v, outs, sample_fn, max_iters=150):
    avg_v = th.zeros_like(v.data)
    for i_update in range(max_iters):
        samples = sample_fn()
        diffs = th.sum((outs.unsqueeze(dim=1) - samples.unsqueeze(dim=0)) ** 2, dim=2)
        min_diffs, _ = th.min(diffs - v.unsqueeze(1), dim=0)
        loss = -(th.mean(min_diffs) + th.mean(v))
        optim_v.zero_grad()
        loss.backward()
        optim_v.step()
        v.data = v.data - th.mean(v.data)
        k = i_update + 1
        avg_v = ((k-1) / k) * avg_v + (1/k) * v.data
    return i_update,  avg_v


def optimize_v_adaptively(outs, v, sample_fn_opt, sample_fn_bin_dev,
                          bin_dev_threshold,
                          bin_dev_iters, init_lr=None, v_opt_iters=25,
                          repeat_iters=10):
    # Optimize V
    n_updates_total = 0
    outs = outs.detach()
    gauss_samples = sample_fn_bin_dev()
    diffs = th.sum((outs.unsqueeze(dim=1) - gauss_samples.unsqueeze(dim=0)) ** 2, dim=2)
    if init_lr is None:
        init_lr = float(var_to_np(th.mean(th.min(diffs, dim=1)[0]))[0] * len(outs) / 50)
    optim_v_orig = th.optim.SGD([v], lr=init_lr)
    optim_v = ScheduledOptimizer(DivideSqrtUpdates(), optim_v_orig, True)

    i_updates, avg_v = optimize_v_optimizer(
      v, optim_v, outs.detach(), sample_fn_opt, max_iters=v_opt_iters)
    v.data = avg_v
    n_updates_total += i_updates + 1
    for _ in range(repeat_iters):
        v.data = avg_v
        bincounts = sample_match_and_bincount(outs, v, sample_fn_bin_dev, iters=bin_dev_iters)
        bin_dev = np.mean(np.abs(bincounts - np.mean(bincounts)))
        if bin_dev < bin_dev_threshold:
            break
        i_updates, avg_v = optimize_v_optimizer(
            v, optim_v, outs.detach(), sample_fn_opt, max_iters=v_opt_iters)
        n_updates_total += i_updates + 1
        v.data = avg_v
    if repeat_iters == 0:
        v.data = avg_v
        bincounts = sample_match_and_bincount(outs, v, sample_fn_bin_dev, iters=bin_dev_iters)
        bin_dev = np.mean(np.abs(bincounts - np.mean(bincounts)))

    return bincounts, bin_dev, n_updates_total


def optimize_v(outs_main, means_per_cluster, stds_per_cluster, v, i_class,
               n_wanted_stds, norm_std_to):
    means_reduced, stds_reduced, largest_stds = reduce_dims_to_large_stds(
        means_per_cluster, stds_per_cluster, i_class, n_wanted_stds, norm_std_to=norm_std_to)
    outs_reduced = outs_main.index_select(dim=1, index=largest_stds)
    sample_fn_dt = lambda: get_gauss_samples(
        len(outs_reduced) * 1, means_reduced.detach(), stds_reduced.detach())
    bincounts, bin_dev, n_updates = optimize_v_adaptively(
        outs_reduced.detach(), v, sample_fn_dt, sample_fn_dt,
        bin_dev_threshold=0.75, bin_dev_iters=5)
    return bincounts, bin_dev, n_updates


def reduce_dims_to_large_stds(means_per_cluster, stds_per_cluster, i_class,
                              n_wanted_stds, norm_std_to):
    this_std = stds_per_cluster[i_class] * stds_per_cluster[i_class]
    _,i_std_sorted = th.sort(this_std)
    largest_stds = i_std_sorted[-n_wanted_stds:]
    if norm_std_to is not None:
        this_std = norm_std_to * this_std / th.sum(this_std)
    stds_reduced = this_std[largest_stds]
    means_reduced = means_per_cluster[i_class][largest_stds]
    return means_reduced, stds_reduced, largest_stds
