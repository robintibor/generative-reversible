import torch as th
import numpy as np


def get_gauss_samples(n_samples, mean, std):
    if mean.is_cuda:
        orig_samples = th.cuda.FloatTensor(n_samples, len(mean)).normal_(0, 1)
    else:
        orig_samples = th.FloatTensor(n_samples, len(mean)).normal_(0, 1)
    orig_samples = th.autograd.Variable(orig_samples)
    samples = (orig_samples * std.unsqueeze(0)) + mean.unsqueeze(0)
    return samples


def sample_mixture_gaussian(sizes_per_cluster, means_per_dim, stds_per_dim):
    # assume mean/std are clusters x dims
    parts = []
    n_dims = means_per_dim.size()[1]
    for n_samples, mean, std in zip(sizes_per_cluster, means_per_dim,
                                    stds_per_dim):
        if n_samples == 0: continue
        assert n_samples > 0
        samples = th.randn(n_samples, n_dims)
        samples = th.autograd.Variable(samples)
        if std.is_cuda:
            samples = samples.cuda()
        samples = samples * std.unsqueeze(0) + mean.unsqueeze(0)
        parts.append(samples)
    all_samples = th.cat(parts, dim=0)
    return all_samples


def sizes_from_weights(size, weights, ):
    weights = weights / np.sum(weights)
    fractional_sizes = weights * size

    rounded = np.int64(np.round(fractional_sizes))
    diff_with_half = (fractional_sizes % 1) - 0.5

    n_total = np.sum(rounded)
    # Those closest to 0.5 rounded, take next biggest or next smallest number
    # to match wanted overall size
    while n_total > size:
        mask = (diff_with_half > 0) & (rounded > 0)
        if np.sum(mask) == 0:
            mask = rounded > 0
        i_min = np.argsort(diff_with_half[mask])[0]
        i_min = np.flatnonzero(mask)[i_min]
        diff_with_half[i_min] += 0.5
        rounded[i_min] -= 1
        n_total -= 1
    while n_total < size:
        mask = (diff_with_half < 0) & (rounded > 0)
        if np.sum(mask) == 0:
            mask = rounded > 0
        i_min = np.argsort(-diff_with_half[mask])[0]
        i_min = np.flatnonzero(mask)[i_min]
        diff_with_half[i_min] -= 0.5
        rounded[i_min] += 1
        n_total += 1

    assert np.sum(rounded) == size
    # pytorch needs list of int
    sizes = [int(s) for s in rounded]
    return sizes


def standard_gaussian_icdf(cdfs):
    # see https://en.wikipedia.org/wiki/Normal_distribution#Quantile_function
    return th.erfinv(2 * cdfs - 1) * np.sqrt(2.0)


def standard_gaussian_cdf(vals):
    #see https://en.wikipedia.org/wiki/Normal_distribution#Cumulative_distribution_function
    return 0.5 * (1 + th.erf(vals / (np.sqrt(2))))


def transform_gaussian_by_dirs(means, stds, directions):
    # directions is directions x dims
    # means is clusters x dims
    # stds is clusters x dims
    transformed_means = th.mm(means, directions.transpose(1, 0)).transpose(1, 0)
    # transformed_means is now
    # directions x clusters
    stds_for_dirs = stds.transpose(1, 0).unsqueeze(0)  # 1 x dims x clusters
    transformed_stds = th.sqrt(th.sum(
        (directions * directions).unsqueeze(2) *
        (stds_for_dirs * stds_for_dirs),
        dim=1))
    # transformed_stds is now
    # directions x clusters
    return transformed_means, transformed_stds


def compute_all_i_cdfs(this_means, this_stds, sorted_weights, directions):
    transformed_means, transformed_stds = transform_gaussian_by_dirs(
        this_means, th.abs(this_stds), directions)

    n_virtual_samples = th.sum(sorted_weights[:, 0])
    start = 1 / (2 * n_virtual_samples)
    wanted_sum = 1 - (2 / (n_virtual_samples))
    probs = sorted_weights * wanted_sum / n_virtual_samples
    empirical_cdf = start + th.cumsum(probs, dim=0)

    # see https://en.wikipedia.orsorted_softmaxedg/wiki/Normal_distribution -> Quantile function
    sqrt_2 = th.autograd.Variable(th.FloatTensor([np.sqrt(2.0)]))
    sqrt_2, empirical_cdf = ensure_on_same_device(sqrt_2, empirical_cdf)
    i_cdf = sqrt_2 * th.erfinv(2 * empirical_cdf - 1)
    i_cdf = i_cdf.squeeze()
    all_i_cdfs = i_cdf * transformed_stds.t() + transformed_means.t()
    return all_i_cdfs
