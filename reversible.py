import torch as th
import torch.nn as nn
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib import cm

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
            previous_features = th.autograd.Variable(
                th.zeros(features.size()[0],
                         features.size()[1] // (module.stride[0] * module.stride[1]),
                         features.size()[2] * module.stride[0],
                         features.size()[3] * module.stride[1]).cuda())

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
    dist_X_X = greedy_min_dist_pair_diff(X, X,
                                         n_iterations=n_iterations,
                                         distfunc=distfunc,
                                         add_unmatched_diffs=add_unmatched_diffs,
                                         remove_diagonal=True)
    dist_Y_Y = greedy_min_dist_pair_diff(Y, Y,
                                         n_iterations=n_iterations,
                                         distfunc=distfunc,
                                         add_unmatched_diffs=add_unmatched_diffs,
                                         remove_diagonal=True)
    loss = 2 * dist_X_Y - dist_Y_Y - dist_X_X
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
        diffs = th.masked_select(diffs, mask).view(n_new_elements,
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
