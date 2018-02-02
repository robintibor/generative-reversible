import torch as th
import torch.nn as nn
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib import cm
from braindecode.torch_ext.util import var_to_np

class ReversibleBlock(th.nn.Module):
    def __init__(self, F, G):
        super(ReversibleBlock, self).__init__()
        self.F = F
        self.G = G

    def forward(self, x):
        n_chans = x.size()[1]
        assert n_chans % 2 == 0
        x1 = x[:, :n_chans // 2]
        x2 = x[:, n_chans // 2:]
        y1 = self.F(x1) + x2
        y2 = self.G(y1) + x1
        return th.cat((y1, y2), dim=1)


class SubsampleSplitter(th.nn.Module):
    def __init__(self, stride):
        super(SubsampleSplitter, self).__init__()
        if not hasattr(stride, '__len__'):
            stride = (stride, stride)
        self.stride = stride

    def forward(self, x):
        new_x = []
        for i_stride in range(self.stride[0]):
            for j_stride in range(self.stride[1]):
                new_x.append(
                    x[:, :, i_stride::self.stride[0], j_stride::self.stride[1]])
        new_x = th.cat(new_x, dim=1)
        return new_x


def invert(feature_model, features):
    if feature_model.__class__.__name__ == 'ReversibleBlock':
        feature_model = nn.Sequential(feature_model, )
    for module in reversed(list(feature_model.children())):
        if module.__class__.__name__ == 'ReversibleBlock':
            n_chans = features.size()[1]
            # y1 = self.F(x1) + x2
            # y2 = self.G(y1) + x1
            y1 = features[:, :n_chans // 2]
            y2 = features[:, n_chans // 2:]

            x1 = y2 - module.G(y1)
            x2 = y1 - module.F(x1)
            features = th.cat((x1, x2), dim=1)
        if module.__class__.__name__ == 'SubsampleSplitter':
            # for i_stride in range(self.stride):
            #    for j_stride in range(self.stride):
            #        new_x.append(x[:,:,i_stride::self.stride, j_stride::self.stride])
            previous_features = th.zeros(features.size()[0],
                         features.size()[1] // (module.stride[0] * module.stride[1]),
                         features.size()[2] * module.stride[0],
                         features.size()[3] * module.stride[1])
            if y1.is_cuda:
                previous_features = previous_features.cuda()
            previous_features = th.autograd.Variable(previous_features)

            n_chans_before = previous_features.size()[1]
            cur_chan = 0
            for i_stride in range(module.stride[0]):
                for j_stride in range(module.stride[1]):
                    previous_features[:, :, i_stride::module.stride[0],
                    j_stride::module.stride[1]] = (
                        features[:,
                        cur_chan * n_chans_before:cur_chan * n_chans_before + n_chans_before])
                    cur_chan += 1
            features = previous_features
    return features


def min_l1_dist_per_cluster(variables, uniform_samples):
    diffs = pairwise_l1_dist(variables, uniform_samples)
    # variables x uniform samples
    min_diffs_per_sample, _ = th.min(diffs, dim=0)
    return min_diffs_per_sample


def pairwise_l1_dist(variables, uniform_samples):
    diffs = variables.unsqueeze(1) - uniform_samples.unsqueeze(0)
    # variables x uniform samples x dims
    diffs = th.mean(th.abs(diffs), dim=2)
    return diffs


def pairwise_squared_l2_dist(variables, uniform_samples):
    diffs = variables.unsqueeze(1) - uniform_samples.unsqueeze(0)
    # variables x uniform samples x dims
    diffs = th.sum(diffs * diffs, dim=2)
    return diffs

def pairwise_l2_dist(variables, uniform_samples):
    diffs = pairwise_squared_l2_dist(variables, uniform_samples)
    eps = 1e-6
    return th.sqrt(diffs + eps)

def min_l1_dist_symmetric(variables, uniform_samples, distfunc=pairwise_l1_dist):
    diffs = distfunc(variables, uniform_samples)
    # variables x uniform samples
    min_diffs_per_sample, _ = th.min(diffs, dim=0)
    min_diffs_per_variable, _ = th.min(diffs, dim=1)
    overall_diff = th.mean(min_diffs_per_sample) + th.mean(
        min_diffs_per_variable)
    return overall_diff

def pairwise_diff_without_diagonal(X, distfunc):
    diffs_X_X = distfunc(X,X)
    eye_mat = th.autograd.Variable(th.eye(diffs_X_X.size()[0]),
                                   requires_grad=False)
    if diffs_X_X.is_cuda:
        eye_mat = eye_mat.cuda()
    diffs_X_X = eye_mat * th.max(diffs_X_X) + diffs_X_X
    return diffs_X_X


def min_diff_diff_loss(X, Y, distfunc):
    diffs = distfunc(X, Y)
    # variables x uniform samples
    min_diffs_XY, _ = th.min(diffs, dim=0)
    min_diffs_YX, _ = th.min(diffs, dim=1)

    diffs_X_X = pairwise_diff_without_diagonal(X, distfunc)
    min_diffs_X_X, _ = th.min(diffs_X_X, dim=0)
    diffs_Y_Y = pairwise_diff_without_diagonal(Y, distfunc)
    min_diffs_Y_Y, _ = th.min(diffs_Y_Y, dim=0)

    loss = th.mean(min_diffs_XY) + th.mean(min_diffs_YX) - th.mean(
        min_diffs_X_X) - th.mean(min_diffs_Y_Y)
    return loss


def min_diff_diff_greedy_loss(X,Y, n_iterations, distfunc, add_unmatched_diffs):
    dist_X_Y = greedy_min_dist_pair_diff(X, Y,
                                         n_iterations=n_iterations,
                                         distfunc=distfunc,
                                         add_unmatched_diffs=add_unmatched_diffs)
    #dist_X_X = greedy_min_dist_pair_diff(X, X,
    #                                     n_iterations=n_iterations,
    #                                     distfunc=distfunc,
    #                                     add_unmatched_diffs=add_unmatched_diffs,
    #                                     remove_diagonal=True)
    #dist_Y_Y = greedy_min_dist_pair_diff(Y, Y,
    #                                     n_iterations=n_iterations,
    #                                     distfunc=distfunc,
    #                                     add_unmatched_diffs=add_unmatched_diffs,
    #                                     remove_diagonal=True)
    n_half = len(X)//2
    dist_X_X = greedy_min_dist_pair_diff(X[:n_half], X[n_half:],
                                        n_iterations=n_iterations,
                                        distfunc=distfunc,
                                        add_unmatched_diffs=add_unmatched_diffs,
                                        remove_diagonal=True)
    dist_Y_Y = greedy_min_dist_pair_diff(Y[:n_half], Y[n_half:],
                                        n_iterations=n_iterations,
                                        distfunc=distfunc,
                                        add_unmatched_diffs=add_unmatched_diffs,
                                        remove_diagonal=True)
    #loss = 2 * dist_X_Y - dist_Y_Y - dist_X_X
    loss = dist_X_Y# / dist_X_X
    return loss


def plot_xyz(x, y, z):
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()
    xx = np.linspace(min(x), max(x), 100)
    yy = np.linspace(min(y), max(y), 100)
    f = interpolate.NearestNDInterpolator(list(zip(x, y)), z)
    assert len(xx) == len(yy)
    zz = np.ones((len(xx), len(yy)))
    for i_x in range(len(xx)):
        for i_y in range(len(yy)):
            # somehow this is correct. don't know why :(
            zz[i_y, i_x] = f(xx[i_x], yy[i_y])
    assert not np.any(np.isnan(zz))

    ax.imshow(zz, vmin=-np.max(np.abs(z)), vmax=np.max(np.abs(z)), cmap=cm.PRGn,
              extent=[min(x), max(x), min(y), max(y)], origin='lower',
              interpolation='nearest', aspect='auto')


def greedy_unique_diff_matrix(diffs, n_iterations, add_unmatched_diffs):
    iteration = 0
    n_elements = th.autograd.Variable(th.zeros(1))
    if diffs.is_cuda:
        n_elements = n_elements.cuda()
    finished_completely = False
    diff_sum = th.autograd.Variable(th.zeros(1))
    if diffs.is_cuda:
        diff_sum = diff_sum.cuda()
    while iteration < n_iterations:
        # min_var_inds -> i_cluster_to_min_var
        # min_cluster_inds -> i_var_to_min_cluster
        mins_per_cluster, min_var_inds = th.min(diffs, dim=0)
        mins_per_var, min_cluster_inds = th.min(diffs, dim=1)
        ind_range = th.autograd.Variable(
            th.arange(0, len(min_cluster_inds)).type(th.LongTensor))
        if min_cluster_inds.is_cuda:
            ind_range = ind_range.cuda()
        i_correct_vars = min_var_inds[min_cluster_inds] == ind_range
        i_correct_clusters = min_cluster_inds[min_var_inds] == ind_range
        diff_sum += th.sum(mins_per_var[i_correct_vars])
        n_this_elements = th.sum(i_correct_vars).type(th.FloatTensor)
        if diffs.is_cuda:
            n_this_elements = n_this_elements.cuda()
        n_elements += n_this_elements
        i_incorrect_vars = (i_correct_vars ^ 1)
        i_incorrect_clusters = (i_correct_clusters ^ 1)
        if (th.sum(i_incorrect_vars) == 0).data.all():
            finished_completely = True
            break
        n_new_elements = th.sum(i_incorrect_clusters.type(th.LongTensor)).data[0]
        # fast way to select
        i_incorrect_vars_float = i_incorrect_vars.type(th.FloatTensor)
        i_incorrect_clusters_float = i_incorrect_clusters.type(th.FloatTensor)
        if diffs.is_cuda:
            i_incorrect_vars_float = i_incorrect_vars_float.cuda()
            i_incorrect_clusters_float = i_incorrect_clusters_float.cuda()
        mask = (i_incorrect_vars_float.unsqueeze(1) *
                 i_incorrect_clusters_float.unsqueeze(0)) == 1
        diffs = th.masked_select(diffs.clone(), mask).view(n_new_elements,
                                                   n_new_elements)
        iteration += 1

    if (not finished_completely) and (add_unmatched_diffs):
        remaining_mins, _ = th.min(diffs, dim=1)
        diff_sum += th.sum(remaining_mins)
        n_elements += remaining_mins.size()[0]
        remaining_mins, _ = th.min(diffs, dim=0)
        diff_sum += th.sum(remaining_mins)
        n_elements += remaining_mins.size()[0]
    mean_diff = diff_sum / n_elements
    return mean_diff


def greedy_min_dist_pair_diff(var_a, var_b, n_iterations=3, distfunc=pairwise_l1_dist,
                              add_unmatched_diffs=True, remove_diagonal=False):
    diffs = distfunc(var_a, var_b)
    if remove_diagonal:
        eye_mat = th.autograd.Variable(th.eye(diffs.size()[0]),
                                       requires_grad=False)
        if diffs.is_cuda:
            eye_mat = eye_mat.cuda()
        diffs = eye_mat * th.max(diffs) + diffs
    mean_diff = greedy_unique_diff_matrix(diffs, n_iterations=n_iterations,
                                          add_unmatched_diffs=add_unmatched_diffs)
    return mean_diff

def add_max_to_diagonal(diffs):
    eye_mat = th.autograd.Variable(th.eye(diffs.size()[0]),
                                   requires_grad=False)
    if diffs.is_cuda:
        eye_mat = eye_mat.cuda()
    diffs = eye_mat * th.max(diffs) + diffs
    return diffs

def log_kernel_density_estimation(X, x, bandwidth):
    # X are reference points / landmark/kernel points
    # x points to determine densities for
    constant = float(1 / np.sqrt(2 * np.pi))
    diffs = X.unsqueeze(1) - x.unsqueeze(0)
    diffs = diffs / bandwidth.unsqueeze(0).unsqueeze(0)
    exped = constant * th.exp(-(diffs * diffs) / 2)
    pdf_per_dim = th.mean(exped, dim=0) / bandwidth.unsqueeze(0)
    eps = 1e-6
    log_kernel_densities = th.sum(th.log(pdf_per_dim + eps), dim=1)
    return log_kernel_densities


def log_gaussian_pdf(x, std):
    #assume x is examples x dim
    # assume std is dim
    #https://www.statlect.com/fundamentals-of-statistics/normal-distribution-maximum-likelihood#hid4
    std = std.unsqueeze(0)
    eps = 1e-6
    subtractor = (-float(np.log(2 * np.pi) / 2) - th.log(std + eps))
    log_pdf_per_dim = (-(x * x) / (std * std*2)) + subtractor
    eps = 1e-6
    log_pdf = th.sum(log_pdf_per_dim + eps, dim=1)
    return log_pdf


def rand_transport_loss(variables, directions, stds, expected_vals):
    directions = sample_directions(variables.size()[1], orthogonalize=True,
                                cuda=variables.is_cuda)
    var_out = th.mm(variables, directions.transpose(1, 0))
    var_out, _ = th.sort(var_out, dim=0)
    diffs = var_out - expected_vals.unsqueeze(1)
    total_loss = th.mean(th.abs(diffs))
    return total_loss

def rand_transport_loss_sampled(variables, directions, stds):
    directions = sample_directions(variables.size()[1], orthogonalize=True,
                                cuda=variables.is_cuda)
    # norm for std
    samples = th.autograd.Variable(th.randn(variables.size())) * stds.unsqueeze(0)
    directed_samples = th.mm(samples, directions.transpose(1, 0))
    sorted_samples, _ = th.sort(directed_samples, dim=0)
    var_out = th.mm(variables, directions.transpose(1, 0))
    var_out, _ = th.sort(var_out, dim=0)
    diffs = var_out - sorted_samples
    total_loss = th.mean(th.abs(diffs))
    return total_loss


def sample_mixture_gaussian(sizes_per_cluster, means_per_dim, stds_per_dim):
    # assume mean/std are clusters x dims
    parts = []
    n_dims = means_per_dim.size()[1]
    for n_samples, mean, std  in zip(sizes_per_cluster, means_per_dim, stds_per_dim):
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
    diff_with_half =  (fractional_sizes % 1) - 0.5

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


def get_weights_per_sample(weights_per_cluster, sizes):
    # weighs will be all one in the end
    # however they are normalized by a number that is not
    # backpropagated through, so there will be an appropriate gradient
    # through them
    all_weights = []
    for i_cluster in range(len(sizes)):
        size = sizes[i_cluster]
        weight = weights_per_cluster[i_cluster]
        if size == 0:
            continue
        assert size > 0
        np_weight = float(var_to_np(weight))
        assert np_weight >= 0
        if np_weight > 0:
            new_weight = weight / np_weight
        else:
            new_weight = weight # should be 0
        all_weights.append(new_weight.expand(size))
    return th.cat(all_weights)


def rand_transport_loss_both_sampled(samples_a, samples_b, weights_b, directions,
                                     abs_or_square):
    weighted_diffs = rand_transport_diffs_both_sampled(samples_a, samples_b, weights_b, directions)
    if abs_or_square == 'abs':
        total_loss = th.mean(th.abs(weighted_diffs))
    else:
        assert abs_or_square == 'square'
        total_loss = th.mean(weighted_diffs * weighted_diffs)
    return total_loss


def rand_transport_diffs_both_sampled(samples_a, samples_b, weights_b, directions):
    if th.is_tensor(directions):
        directions = th.autograd.Variable(directions, requires_grad=False)
    norm_factors = th.norm(directions, p=2, dim=1, keepdim=True)
    directions = directions / norm_factors
    directed_samples_a = th.mm(samples_a, directions.transpose(1, 0))
    sorted_samples_a, _ = th.sort(directed_samples_a, dim=0)
    directed_samples_b = th.mm(samples_b, directions.transpose(1, 0))
    sorted_samples_b, sort_inds = th.sort(directed_samples_b, dim=0)
    diffs = sorted_samples_b - sorted_samples_a
    weighted_diffs = th.cat([(weights_b[sort_inds[:, i_dim]] *
                              diffs[:, i_dim]).unsqueeze(1)
                             for i_dim in range(sort_inds.size()[1])], dim=1)
    return weighted_diffs


def compute_transport_loss(batch_outs, weights_per_cluster, means_per_dim, stds_per_dim,
                           abs_or_square, sample_gauss_repetitions=1):
    batch_size = len(batch_outs)
    directions = sample_directions(batch_outs.size()[1], orthogonalize=True,
                                cuda=batch_outs.is_cuda)
    if sample_gauss_repetitions == 1:
        sizes = sizes_from_weights(batch_size, var_to_np(weights_per_cluster))
        normed_weights = weights_per_cluster / th.sum(weights_per_cluster)
        weights_per_sample = get_weights_per_sample(normed_weights, sizes)
        samples = sample_mixture_gaussian(sizes, means_per_dim, stds_per_dim)
        this_transport_loss = rand_transport_loss_both_sampled(batch_outs, samples, weights_per_sample,
                                    directions, abs_or_square)
    else:
        this_transport_loss = repeated_rand_transport_diffs_both_sampled(
            batch_outs, weights_per_cluster, means_per_dim, stds_per_dim,
            directions, abs_or_square, sample_gauss_repetitions=sample_gauss_repetitions)
    return this_transport_loss


def norm_and_var_directions(directions):
    if th.is_tensor(directions):
        directions = th.autograd.Variable(directions, requires_grad=False)
    norm_factors = th.norm(directions, p=2, dim=1, keepdim=True)
    directions = directions / norm_factors
    return directions

def repeated_rand_transport_diffs_both_sampled(batch_outs, weights_per_cluster, means_per_dim, stds_per_dim,
                           directions, abs_or_square, sample_gauss_repetitions=1):
    directions = norm_and_var_directions(directions)
    directed_samples_batch = th.mm(batch_outs, directions.transpose(1, 0))
    sorted_samples_batch, _ = th.sort(directed_samples_batch, dim=0)
    sizes = sizes_from_weights(len(batch_outs), var_to_np(weights_per_cluster))
    normed_weights = weights_per_cluster / th.sum(weights_per_cluster)
    weights_per_sample = get_weights_per_sample(normed_weights, sizes)
    all_diffs = []
    for _ in range(sample_gauss_repetitions):
        samples_cluster = sample_mixture_gaussian(sizes, means_per_dim, stds_per_dim)
        directed_samples_cluster = th.mm(samples_cluster, directions.transpose(1, 0))
        sorted_samples_cluster, sort_inds = th.sort(directed_samples_cluster, dim=0)
        diffs = sorted_samples_cluster - sorted_samples_batch
        weighted_diffs = th.cat([(weights_per_sample[sort_inds[:, i_dim]] *
                                  diffs[:, i_dim]).unsqueeze(1)
                                 for i_dim in range(sort_inds.size()[1])],
                                dim=1)
        all_diffs.append(weighted_diffs)
    mean_weighted_diffs = th.mean(th.stack(all_diffs, dim=0), dim=0)
    if abs_or_square == 'abs':
        total_loss = th.mean(th.abs(mean_weighted_diffs))
    else:
        assert abs_or_square == 'square'
        total_loss = th.mean(mean_weighted_diffs * mean_weighted_diffs)
    return total_loss

# see
# https://stats.stackexchange.com/questions/187828/how-are-the-error-function-and-standard-normal-distribution-function-related
# and https://en.wikipedia.org/wiki/Normal_distribution#Cumulative_distribution_function
def multi_gaussian_cdf(x, means, stds, weights):
    cdfs = (1 + th.erf((x.unsqueeze(1)-means.unsqueeze(0))/(stds.unsqueeze(0)*np.sqrt(2)))) / 2.0
    # examples x mixture components
    cdf = th.sum(cdfs * weights.unsqueeze(0), dim=1)
    return cdf


def compute_sample_points_gaussian_mixture(means_per_dim, stds_per_dim,
                                           weights_per_cluster,
                                           n_samples, n_interpolation_points):
    start = float(th.min(means_per_dim - stds_per_dim * 4).data)
    stop = float(th.max(means_per_dim + stds_per_dim * 4).data)
    x = th.linspace(start, stop, n_interpolation_points)
    x = th.autograd.Variable(x)
    cdf_of_x = multi_gaussian_cdf(x, means_per_dim.squeeze(1),
                                  stds_per_dim.squeeze(1),
                                  weights_per_cluster / th.sum(
                                      weights_per_cluster))
    wanted_probs = th.autograd.Variable(
        th.linspace(1 / n_samples, 1 - 1 / n_samples, n_samples))

    mask = cdf_of_x.unsqueeze(0) < wanted_probs.unsqueeze(1)

    _, min_inds = th.min(mask, dim=1)

    sample_points = x[min_inds]
    return sample_points


def multi_directions_gaussian_cdfs(x, means, stds, weights):
    ## and https://en.wikipedia.org/wiki/Normal_distribution#Cumulative_distribution_function
    # assuming input x is directions x 1-dimensional points
    # assuming means/stds are directions x clusters
    # weights is 1-dimension (number of clusters)
    weights = weights / th.sum(weights)
    cdfs = 0.5 * (1 +
            th.erf((x.unsqueeze(2) - means.unsqueeze(1)) / (
                stds.unsqueeze(1) * np.sqrt(2))))
    # directions x points x clusters
    cdf = th.sum(cdfs * weights.unsqueeze(0).unsqueeze(1), dim=2)
    # directions x points
    return cdf


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


def compute_multi_dir_sample_points_gaussian_mixture(dir_means, dir_stds,
                                           weights_per_cluster,
                                           n_samples, n_interpolation_points):
    # means stds are
    # directions x clusters
    # first create points where to interpolate cdf
    all_x = []
    for this_mean, this_std in zip(dir_means, dir_stds):
        start = float(th.min(this_mean - this_std * 4).data)
        stop = float(th.max(this_mean + this_std * 4).data)
        x = th.linspace(start, stop, n_interpolation_points)
        if dir_means.is_cuda:
            x = x.cuda()
        all_x.append(x)

    x = th.stack(all_x, dim=0)
    x = th.autograd.Variable(x)
    # directions x interpolation points


    cdfs_of_x = multi_directions_gaussian_cdfs(x, dir_means, dir_stds,
                                               weights_per_cluster / th.sum(
                                                   weights_per_cluster))
    # directions x interpolation points

    # Compute inverse cdf at those probabilities
    # (Assuming delta distribution as pdf and corresponding cdf
    # for samples)
    wanted_probs = th.linspace(1 / n_samples, 1 - 1 / n_samples, n_samples)
    if dir_means.is_cuda:
        wanted_probs = wanted_probs.cuda()
    wanted_probs = th.autograd.Variable(wanted_probs)

    # find minima for each probability
    # for each direction
    mask = cdfs_of_x.unsqueeze(0) < wanted_probs.unsqueeze(1).unsqueeze(1)
    # now wanted probs x directions x original interpolation points
    # So find minimum over interpolation points
    _, min_inds = th.min(mask, dim=2)

    # Loop through directions, collect interpolation points
    all_sample_points = []
    for i_direction in range(len(x)):
        all_sample_points.append(x[i_direction][min_inds[:, i_direction]])
    all_sample_points = th.stack(all_sample_points, dim=0)
    return all_sample_points

def sample_directions(n_dims, orthogonalize, cuda):
    directions = th.randn(n_dims, n_dims)
    if orthogonalize:
        directions, _ = th.qr(directions)

    if cuda:
        directions = directions.cuda()
    directions = th.autograd.Variable(directions, requires_grad=False)
    norm_factors = th.norm(directions, p=2, dim=1, keepdim=True)
    directions = directions / norm_factors
    return directions

def symmetric_analytic_transport_loss(batch_outs, means_per_dim, stds_per_dim,
                                     weights_per_cluster,
                                     n_interpolation_points):
    directions = sample_directions(batch_outs.size()[1], orthogonalize=True,
                                cuda=batch_outs.is_cuda)

    projected_outs = th.mm(batch_outs, directions.transpose(1, 0)).transpose(1,0)
    # now directions x outs

    dir_means, dir_stds = transform_gaussian_by_dirs(
        means_per_dim, stds_per_dim, directions)

    sorted_outs, _ = th.sort(projected_outs, dim=1)
    wanted_cdfs = multi_directions_gaussian_cdfs(sorted_outs, dir_means, dir_stds,
                       weights_per_cluster / th.sum(weights_per_cluster))

    inter_points = wanted_cdfs * (len(batch_outs)- 1)
    # crashed once, maybe because of numerical issue?
    inter_points = th.clamp(inter_points, min=0, max=len(batch_outs) - 1)
    i_assigned_1 = th.floor(inter_points)
    i_assigned_2 = th.ceil(inter_points)
    weight_per_assignment = i_assigned_2 - inter_points
    # still directions x points/outs

    gaussian_cdf_to_points_losses = []
    # Loop over directions
    for this_weights, this_ins, this_assigned_1, this_assigned_2 in zip(
            weight_per_assignment, sorted_outs, i_assigned_1, i_assigned_2):
        assigned_1_data = this_assigned_1.data
        assigned_2_data = this_assigned_2.data
        if this_ins.is_cuda:
            assigned_1_data = assigned_1_data.type(th.cuda.LongTensor)
            assigned_2_data = assigned_2_data.type(th.cuda.LongTensor)
        else:
            assigned_1_data = assigned_1_data.type(th.LongTensor)
            assigned_2_data = assigned_2_data.type(th.LongTensor)

        assigned_inputs = this_weights.unsqueeze(1) * this_ins[assigned_1_data] + (
            (1 - this_weights)).unsqueeze(1) * this_ins[assigned_2_data]
        diffs = th.autograd.Variable(this_ins.data) - assigned_inputs
        loss = th.mean(diffs * diffs)
        gaussian_cdf_to_points_losses.append(loss)


    n_samples = len(batch_outs)
    all_sample_points = compute_multi_dir_sample_points_gaussian_mixture(dir_means, dir_stds,
                                           weights_per_cluster,
                                           n_samples, n_interpolation_points)
    diffs = all_sample_points - sorted_outs
    empirical_cdf_to_gaussian_loss = th.mean(diffs * diffs)
    loss = th.mean(th.cat(gaussian_cdf_to_points_losses)) + empirical_cdf_to_gaussian_loss
    return loss


def kl_cum_div(a_cdf, b_cdf, mean_a, mean_b):
    eps = 1e-6
    #http://onlinelibrary.wiley.com/doi/10.1002/asmb.2116/full
    return th.mean(a_cdf * (th.log(a_cdf+ eps) - th.log(b_cdf + eps))) + th.mean(th.abs(mean_a - mean_b))


def symmetric_kl_cum_div(a_cdf, b_cdf, mean_a, mean_b):
    return kl_cum_div(a_cdf, b_cdf, mean_a, mean_b) + kl_cum_div(b_cdf, a_cdf, mean_b, mean_a)


def analytic_kl_div(samples, means_per_dim, stds_per_dim,
                                     weights_per_cluster):
    directions = sample_directions(n_dims=samples.size()[1], orthogonalize=True, cuda=samples.is_cuda)
    analytical_cdf, empirical_cdf, analytical_means, empirical_means = analytical_empirical_cdf(
        samples, directions, means_per_dim, stds_per_dim, weights_per_cluster)
    kl_div = symmetric_kl_cum_div(analytical_cdf,empirical_cdf, analytical_means, empirical_means)
    return kl_div


def analytical_empirical_cdf(
        samples, directions, means_per_dim, stds_per_dim, weights_per_cluster):
    projected_samples = th.mm(samples, directions.t())
    sorted_samples, _ = th.sort(projected_samples, dim=0)
    mean_dirs, std_dirs = transform_gaussian_by_dirs(means_per_dim,
                                                     stds_per_dim, directions)

    assert (weights_per_cluster >= 0).data.all()
    normed_weights = weights_per_cluster / th.sum(weights_per_cluster)
    analytical_cdf = multi_directions_gaussian_cdfs(sorted_samples.t(),
                                                    mean_dirs, std_dirs,
                                                    normed_weights)
    analytical_means = th.sum(mean_dirs * normed_weights, dim=1)

    empirical_means = th.mean(sorted_samples, dim=0)

    n_samples = len(sorted_samples)
    empirical_cdf = th.linspace(1 / n_samples, 1 - 1 / n_samples, n_samples).unsqueeze(0)
    empirical_cdf = th.autograd.Variable(empirical_cdf)
    return analytical_cdf, empirical_cdf, analytical_means, empirical_means


def analytical_l2_cdf_loss(samples, means_per_dim, stds_per_dim,
                                     weights_per_cluster, directions=None):
    if directions is None:
        directions = sample_directions(n_dims=samples.size()[1],
                                       orthogonalize=True, cuda=samples.is_cuda)
    analytical_cdf, empirical_cdf, analytical_means, empirical_means = analytical_empirical_cdf(
            samples, directions, means_per_dim, stds_per_dim, weights_per_cluster)

    diffs = analytical_cdf - empirical_cdf
    return th.mean(th.sqrt(th.sum(diffs * diffs, dim=1)))




def analytical_komolgorov_loss(samples, means_per_dim, stds_per_dim,
                                     weights_per_cluster):
    directions = sample_directions(n_dims=samples.size()[1], orthogonalize=True, cuda=samples.is_cuda)
    analytical_cdf, empirical_cdf, analytical_means, empirical_means = analytical_empirical_cdf(
            samples, directions, means_per_dim, stds_per_dim, weights_per_cluster)

    diffs = analytical_cdf - empirical_cdf
    return th.max(diffs * diffs)


def ensure_on_same_device(*variables):
    any_cuda = np.any([v.is_cuda for v in variables])
    if any_cuda:
        variables = [ensure_cuda(v) for v in variables]
    return variables


def ensure_cuda(v):
    if not v.is_cuda:
        v = v.cuda()
    return v


def projected_samples_mixture_sorted(weights_per_cluster, means_per_dim,
                                     stds_per_dim,
                                     directions, n_samples,
                                     n_interpolation_samples):
    sizes = sizes_from_weights(n_interpolation_samples,
                               var_to_np(weights_per_cluster))
    dir_means, dir_stds = transform_gaussian_by_dirs(means_per_dim,
                                                     stds_per_dim, directions)
    cluster_samples = sample_mixture_gaussian(sizes, dir_means.t(),
                                              dir_stds.t())
    sorted_cluster_samples, _ = th.sort(cluster_samples, dim=0)
    offset_x_in_input = -0.5 + 0.5 * (
        len(sorted_cluster_samples) / n_samples)
    x_grid = th.linspace(offset_x_in_input,
                         len(sorted_cluster_samples) - 1 - offset_x_in_input,
                         n_samples)
    i_low = th.floor(x_grid)
    i_high = th.ceil(x_grid)
    weights_high = x_grid - i_low
    i_low = th.clamp(i_low, min=0)
    i_high = th.clamp(i_high, max=n_interpolation_samples-1)
    i_low = i_low.type(th.LongTensor)
    i_high = i_high.type(th.LongTensor)
    weights_high = th.autograd.Variable(weights_high)
    i_low, i_high, sorted_cluster_samples, weights_high = ensure_on_same_device(
        i_low, i_high, sorted_cluster_samples, weights_high)
    vals_low = sorted_cluster_samples[i_low]
    vals_high = sorted_cluster_samples[i_high]
    vals_interpolated = (vals_low * (1 - weights_high).unsqueeze(
        1)) + (vals_high * weights_high.unsqueeze(1))
    return vals_interpolated


def sampled_transport_diffs_interpolate_sorted(batch_outs, weights_per_cluster,
                                               means_per_dim, stds_per_dim,
                                               directions, abs_or_square,
                                               n_interpolation_samples):
    directions = norm_and_var_directions(directions)
    directed_samples_batch = th.mm(batch_outs, directions.transpose(1, 0))
    sorted_samples_batch, _ = th.sort(directed_samples_batch, dim=0)
    sorted_samples_cluster = projected_samples_mixture_sorted(
        weights_per_cluster, means_per_dim, stds_per_dim,
        directions, len(batch_outs),
        n_interpolation_samples=n_interpolation_samples)
    diffs = sorted_samples_cluster - sorted_samples_batch
    if abs_or_square == 'abs':
        total_loss = th.mean(th.abs(diffs))
    else:
        assert abs_or_square == 'square'
        total_loss = th.mean(diffs * diffs)
    return total_loss


def analytical_l2_cdf_and_sample_transport_loss(
        samples, means_per_dim, stds_per_dim, weights_per_cluster, abs_or_square,
        n_interpolation_samples,
        cuda=False, directions=None):
    # common
    if directions is None:
        directions = sample_directions(samples.size()[1], True, cuda=cuda)
    else:
        directions = norm_and_var_directions(directions)

    projected_samples = th.mm(samples, directions.t())
    sorted_samples, _ = th.sort(projected_samples, dim=0)
    cdf_loss =  analytical_l2_cdf_loss_given_sorted_samples(
        sorted_samples, directions,
        means_per_dim, stds_per_dim, weights_per_cluster)
    sample_loss =  sampled_transport_diffs_interpolate_sorted_part(
        sorted_samples, directions, means_per_dim,
        stds_per_dim, weights_per_cluster, n_interpolation_samples,
        abs_or_square=abs_or_square)
    return cdf_loss, sample_loss

def analytical_l2_cdf_loss_given_sorted_samples(
        sorted_samples_batch, directions,
        means_per_dim, stds_per_dim, weights_per_cluster):
    n_samples = len(sorted_samples_batch)
    mean_dirs, std_dirs = transform_gaussian_by_dirs(
        means_per_dim, stds_per_dim, directions)
    assert (weights_per_cluster >= 0).data.all()
    normed_weights = weights_per_cluster / th.sum(weights_per_cluster)
    analytical_cdf = multi_directions_gaussian_cdfs(sorted_samples_batch.t(),
                                                    mean_dirs, std_dirs,
                                                    normed_weights)
    empirical_cdf = th.linspace(1 / (n_samples + 1), 1 - (1 / (n_samples+1)),
                                n_samples).unsqueeze(0)
    empirical_cdf = th.autograd.Variable(empirical_cdf)
    directions, empirical_cdf = ensure_on_same_device(directions, empirical_cdf)
    diffs = analytical_cdf - empirical_cdf
    #cdf_loss = th.mean(th.sqrt(th.sum(diffs * diffs, dim=1)))
    cdf_loss = th.mean(th.sqrt(th.mean(diffs * diffs, dim=1)))
    return cdf_loss

def sampled_transport_diffs_interpolate_sorted_part(
        sorted_samples_batch, directions, means_per_dim,
        stds_per_dim, weights_per_cluster, n_interpolation_samples, abs_or_square):
    # sampling based stuff
    sorted_samples_cluster = projected_samples_mixture_sorted(
        weights_per_cluster, means_per_dim, stds_per_dim,
        directions, len(sorted_samples_batch),
        n_interpolation_samples=n_interpolation_samples)
    diffs = sorted_samples_cluster - sorted_samples_batch
    if abs_or_square == 'abs':
        sample_loss = th.mean(th.abs(diffs))
    else:
        assert abs_or_square == 'square'
        sample_loss = th.mean(diffs * diffs)
    return sample_loss
