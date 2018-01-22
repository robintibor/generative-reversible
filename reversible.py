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
        self.stride = stride

    def forward(self, x):
        new_x = []
        for i_stride in range(self.stride):
            for j_stride in range(self.stride):
                new_x.append(
                    x[:, :, i_stride::self.stride, j_stride::self.stride])
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
                         features.size()[1] // (module.stride * module.stride),
                         features.size()[2] * module.stride,
                         features.size()[3] * module.stride).cuda())

            n_chans_before = previous_features.size()[1]
            cur_chan = 0
            for i_stride in range(module.stride):
                for j_stride in range(module.stride):
                    previous_features[:, :, i_stride::module.stride,
                    j_stride::module.stride] = (
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


def min_l1_dist_symmetric(variables, uniform_samples):
    diffs = pairwise_l1_dist(variables, uniform_samples)
    # variables x uniform samples
    min_diffs_per_sample, _ = th.min(diffs, dim=0)
    min_diffs_per_variable, _ = th.min(diffs, dim=1)
    overall_diff = th.mean(min_diffs_per_sample) + th.mean(
        min_diffs_per_variable)
    return overall_diff


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
              interpolation='nearest')


def greedy_unique_diffs(diffs):
    finished = False
    n_iterations = 1
    iteration = 0
    while not finished:
        finished = True
        # assume clusters is in dim 1
        # so var x clusers
        mins, min_var_inds = th.min(diffs, dim=0)
        min_var_to_cluster = dict()
        for cluster, min_var in enumerate(min_var_inds):
            min_var = min_var.data.cpu().numpy()[0]
            if min_var not in min_var_to_cluster:
                min_var_to_cluster[min_var] = []
            min_var_to_cluster[min_var].append(cluster)

        min_vars = list(min_var_to_cluster.keys())
        for min_var in min_vars:
            clusters = min_var_to_cluster[min_var]
            if len(clusters) > 1:
                _, min_cluster = th.min(mins[th.from_numpy(np.array(clusters))],
                                        dim=0)
                min_cluster = clusters[min_cluster.data.cpu().numpy().squeeze()]
                finished = False
            else:
                min_cluster = clusters[0]
            min_diff = diffs[min_var, min_cluster].clone()
            diffs[:, min_cluster] = 100000
            diffs[min_var, :] = 100000
            diffs[min_var, min_cluster] = min_diff
        iteration += 1
        if iteration >= n_iterations:
            break
    return diffs

def greedy_min_dist_diff(var_a, var_b):
    diffs = pairwise_l1_dist(var_a, var_b)
    diffs = greedy_unique_diffs(diffs)
    mins, _ = th.min(diffs,dim=1)
    return th.mean(mins)
