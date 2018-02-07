import torch as th
import torch.nn as nn
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib import cm
from braindecode.torch_ext.util import var_to_np
from braindecode.datautil.iterators import get_balanced_batches


### Reversible model parts

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
    def __init__(self, stride, chunk_chans_first=True):
        super(SubsampleSplitter, self).__init__()
        if not hasattr(stride, '__len__'):
            stride = (stride, stride)
        self.stride = stride
        self.chunk_chans_first = chunk_chans_first

    def forward(self, x):
        # Chunk chans first to ensure that each of the two streams in the
        # reversible network will see a subsampled version of the whole input
        # (in case the preceding blocks would not alter the input)
        # and not one half of the input
        new_x = []
        if self.chunk_chans_first:
            xs = th.chunk(x, 2, dim=1)
        else:
            xs = [x]
        for one_x in xs:
            for i_stride in range(self.stride[0]):
                for j_stride in range(self.stride[1]):
                    new_x.append(
                        one_x[:, :, i_stride::self.stride[0], j_stride::self.stride[1]])
        new_x = th.cat(new_x, dim=1)
        return new_x


def invert(feature_model, features):
    if feature_model.__class__.__name__ == 'ReversibleBlock' or feature_model.__class__.__name__  == 'SubsampleSplitter':
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
            # after splitting the input into two along channel dimension if possible
            # for i_stride in range(self.stride):
            #    for j_stride in range(self.stride):
            #        new_x.append(one_x[:,:,i_stride::self.stride, j_stride::self.stride])
            n_all_chans_before = features.size()[1] // (module.stride[0] * module.stride[1])
            # if ther was only one chan before, chunk had no effect
            if module.chunk_chans_first and (n_all_chans_before > 1):
                chan_features = th.chunk(features, 2, dim=1)
            else:
                chan_features = [features]
            all_previous_features = []
            for one_chan_features in chan_features:
                previous_features = th.zeros(one_chan_features.size()[0],
                             one_chan_features.size()[1] // (module.stride[0] * module.stride[1]),
                             one_chan_features.size()[2] * module.stride[0],
                             one_chan_features.size()[3] * module.stride[1])
                if features.is_cuda:
                    previous_features = previous_features.cuda()
                previous_features = th.autograd.Variable(previous_features)

                n_chans_before = previous_features.size()[1]
                cur_chan = 0
                for i_stride in range(module.stride[0]):
                    for j_stride in range(module.stride[1]):
                        previous_features[:, :, i_stride::module.stride[0],
                        j_stride::module.stride[1]] = (
                            one_chan_features[:,
                            cur_chan * n_chans_before:cur_chan * n_chans_before + n_chans_before])
                        cur_chan += 1
                all_previous_features.append(previous_features)
            features = th.cat(all_previous_features, dim=1)
    return features


def get_inputs_from_reverted_samples(n_inputs, means_per_dim, stds_per_dim,
                                     weights_per_cluster,
                                     feature_model):
    feature_model.eval()
    sizes = sizes_from_weights(n_inputs, var_to_np(weights_per_cluster))
    gauss_samples = sample_mixture_gaussian(sizes, means_per_dim, stds_per_dim)
    rec_var = invert(feature_model, gauss_samples.unsqueeze(2).unsqueeze(3))
    rec_examples = var_to_np(rec_var).squeeze()
    return rec_examples


def weights_init(module, conv_weight_init_fn):
    classname = module.__class__.__name__
    if (('Conv' in classname) or (
        'Linear' in classname)) and classname != "AvgPool2dWithConv":
        conv_weight_init_fn(module.weight)
        if module.bias is not None:
            th.nn.init.constant(module.bias, 0)
    elif 'BatchNorm' in classname:
        th.nn.init.constant(module.weight, 1)
        th.nn.init.constant(module.bias, 0)


def init_model_params(feature_model, gain):
    feature_model.apply(lambda module: weights_init(
        module,
        lambda w: th.nn.init.xavier_uniform(w, gain=gain)))


## Sampling gaussian mixture

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



## Transport/Wasserstein loss

def sample_transport_loss(
        samples, means_per_dim, stds_per_dim, weights_per_cluster, abs_or_square,
        n_interpolation_samples,
        cuda=False, directions=None, backprop_sample_loss_to_cluster_weights=True,
        normalize_by_stds=True, energy_sample_loss=False):
    # common
    if directions is None:
        directions = sample_directions(samples.size()[1], True, cuda=cuda)
    else:
        directions = norm_and_var_directions(directions)

    projected_samples = th.mm(samples, directions.t())
    sorted_samples, _ = th.sort(projected_samples, dim=0)
    if energy_sample_loss:
        sample_loss = sampled_energy_transport_loss(
            projected_samples, directions,
            means_per_dim, stds_per_dim, weights_per_cluster,
            abs_or_square,
            backprop_to_cluster_weights=backprop_sample_loss_to_cluster_weights,
            normalize_by_stds=normalize_by_stds)
    else:
        sample_loss =  sampled_transport_diffs_interpolate_sorted_part(
            sorted_samples, directions, means_per_dim,
            stds_per_dim, weights_per_cluster, n_interpolation_samples,
            abs_or_square=abs_or_square,
            backprop_to_cluster_weights=backprop_sample_loss_to_cluster_weights,
            normalize_by_stds=normalize_by_stds)
    return sample_loss


def sampled_transport_diffs_interpolate_sorted_part(
        sorted_samples_batch, directions, means_per_dim,
        stds_per_dim, weights_per_cluster, n_interpolation_samples, abs_or_square,
        backprop_to_cluster_weights, normalize_by_stds):
    # sampling based stuff
    sorted_samples_cluster, diff_weights, stds_per_sample = projected_samples_mixture_sorted(
        weights_per_cluster, means_per_dim, stds_per_dim,
        directions, len(sorted_samples_batch),
        n_interpolation_samples=n_interpolation_samples,
        backprop_to_cluster_weights=backprop_to_cluster_weights,
        compute_stds_per_sample=normalize_by_stds)
    diffs = sorted_samples_cluster - sorted_samples_batch
    if normalize_by_stds:
        diffs = diffs / stds_per_sample
    else:
        assert stds_per_sample is None

    if abs_or_square == 'abs':
        if backprop_to_cluster_weights:
            sample_loss = th.mean(th.abs(diffs) * diff_weights)
        else:
            sample_loss = th.mean(th.abs(diffs))
    else:
        assert abs_or_square == 'square'
        if backprop_to_cluster_weights:
            sample_loss = th.sqrt(th.mean((diffs * diffs) * diff_weights))
        else:
            sample_loss = th.sqrt(th.mean(diffs * diffs))
    return sample_loss


def sampled_energy_transport_loss(
        projected_samples, directions,
        means_per_dim, stds_per_dim, weights_per_cluster,
        abs_or_square,
        backprop_to_cluster_weights,
        normalize_by_stds):
    permuted_samples = projected_samples[th.randperm(len(projected_samples))]
    proj_samples_a, proj_samples_b = th.chunk(permuted_samples, 2)
    sorted_samples_a, _ = th.sort(proj_samples_a, dim=0)
    sorted_samples_b, _ = th.sort(proj_samples_b, dim=0)
    sorted_samples_cluster_a, diff_weights_a, stds_per_sample_a = projected_samples_mixture_sorted(
        weights_per_cluster, means_per_dim, stds_per_dim,
        directions, len(sorted_samples_a),
        n_interpolation_samples=len(sorted_samples_a),
        backprop_to_cluster_weights=backprop_to_cluster_weights,
        compute_stds_per_sample=normalize_by_stds)
    eps = 1e-6
    if stds_per_sample_a is not None:
        stds_per_sample_a = th.clamp(stds_per_sample_a, min=eps)
    sorted_samples_cluster_b, diff_weights_b, stds_per_sample_b = projected_samples_mixture_sorted(
        weights_per_cluster, means_per_dim, stds_per_dim,
        directions, len(sorted_samples_b),
        n_interpolation_samples=len(sorted_samples_a),
        backprop_to_cluster_weights=backprop_to_cluster_weights,
        compute_stds_per_sample=normalize_by_stds)
    eps = 1e-6
    if stds_per_sample_b is not None:
        stds_per_sample_b = th.clamp(stds_per_sample_b, min=eps)
    diffs_x_y_a = sorted_samples_a - sorted_samples_cluster_a
    diffs_x_y_b = sorted_samples_b - sorted_samples_cluster_b
    diffs_x_x = sorted_samples_a - sorted_samples_b
    diffs_y_y = sorted_samples_cluster_a - sorted_samples_cluster_b


    if normalize_by_stds:
        diffs_x_y_a = diffs_x_y_a / stds_per_sample_a
        diffs_x_y_b = diffs_x_y_b / stds_per_sample_b
        diffs_y_y = diffs_y_y / ((stds_per_sample_a + stds_per_sample_b) / 2)

    if abs_or_square == 'abs':
        diffs_x_x = th.mean(th.abs(diffs_x_x))
        if backprop_to_cluster_weights:
            diffs_x_y_a = th.mean(th.abs(diffs_x_y_a) * diff_weights_a)
            diffs_x_y_b = th.mean(th.abs(diffs_x_y_b) * diff_weights_b)
            diffs_y_y = th.mean(th.abs(diffs_y_y) * ((diff_weights_a + diff_weights_b) / 2))
        else:
            diffs_x_y_a = th.mean(th.abs(diffs_x_y_a))
            diffs_x_y_b = th.mean(th.abs(diffs_x_y_b))
            diffs_y_y = th.mean(th.abs(diffs_y_y))
    else:
        assert abs_or_square == 'square'
        diffs_x_x = th.mean((diffs_x_x * diffs_x_x))
        if backprop_to_cluster_weights:
            diffs_x_y_a = th.mean((diffs_x_y_a * diffs_x_y_a) * diff_weights_a)
            diffs_x_y_b = th.mean((diffs_x_y_b * diffs_x_y_b) * diff_weights_b)
            diffs_y_y = th.mean((diffs_y_y * diffs_y_y) * ((diff_weights_a + diff_weights_b) / 2))
        else:
            diffs_x_y_a = th.mean((diffs_x_y_a * diffs_x_y_a))
            diffs_x_y_b = th.mean((diffs_x_y_b * diffs_x_y_b))
            diffs_y_y = th.mean((diffs_y_y * diffs_y_y))
    #sample_loss = 2 * diffs_x_y_a - diffs_x_x - diffs_y_y
    sample_loss = diffs_x_y_b + diffs_x_y_a - diffs_x_x - diffs_y_y
    sample_loss = sample_loss / diffs_x_x
    return sample_loss


def projected_samples_mixture_sorted(
        weights_per_cluster, means_per_dim, stds_per_dim,
        directions, n_samples, n_interpolation_samples,
        backprop_to_cluster_weights, compute_stds_per_sample):
    sizes = sizes_from_weights(n_interpolation_samples,
                               var_to_np(weights_per_cluster))
    dir_means, dir_stds = transform_gaussian_by_dirs(means_per_dim,
                                                     stds_per_dim, directions)
    cluster_samples = sample_mixture_gaussian(sizes, dir_means.t(),
                                              dir_stds.t())
    if backprop_to_cluster_weights:
        weights_per_sample = get_weights_per_sample(
            weights_per_cluster /th.sum(weights_per_cluster), sizes)
    sorted_cluster_samples, sort_inds = th.sort(cluster_samples, dim=0)
    if backprop_to_cluster_weights:
        weights_per_sample = th.stack(
            [weights_per_sample[sort_inds[:, i_dim]]
             for i_dim in range(sort_inds.size()[1])],
            dim=1)
    if compute_stds_per_sample:
        # these are std factors per sample, unsorted
        std_factors = []
        for i_cluster, size in enumerate(sizes):
            if size > 0:
                std_factors.append(
                    dir_stds[:, i_cluster:i_cluster + 1].repeat(1, size))
        std_factors = th.cat(std_factors, dim=1)
        # now directions x samples
        std_factors = std_factors.t()
        # now samples x directions
        std_per_sample = th.stack(
            [std_factors[:, i_dim][sort_inds[:, i_dim]]
             for i_dim in range(sort_inds.size()[1])],
            dim=1)

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
    if backprop_to_cluster_weights:
        weights_per_sample = (weights_per_sample[i_low] * (1 - weights_high).unsqueeze(1) +
                              weights_per_sample[i_high] * weights_high.unsqueeze(1))
    else:
        weights_per_sample = None
    if compute_stds_per_sample:
        std_per_sample = (std_per_sample[i_low] * (1 - weights_high).unsqueeze(1) +
                              std_per_sample[i_high] * weights_high.unsqueeze(1))
    else:
        std_per_sample = None
    return vals_interpolated, weights_per_sample, std_per_sample


def get_weights_per_sample(weights_per_cluster, sizes):
    # weights will be all one in the end
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


def norm_and_var_directions(directions):
    if th.is_tensor(directions):
        directions = th.autograd.Variable(directions, requires_grad=False)
    norm_factors = th.norm(directions, p=2, dim=1, keepdim=True)
    directions = directions / norm_factors
    return directions


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


def ensure_on_same_device(*variables):
    any_cuda = np.any([v.is_cuda for v in variables])
    if any_cuda:
        variables = [ensure_cuda(v) for v in variables]
    return variables


def ensure_cuda(v):
    if not v.is_cuda:
        v = v.cuda()
    return v


def train_epoch(
        inputs, batch_size, rng,
        feature_model,
        means_per_dim, stds_per_dim, weights_per_cluster,
        directions_adv,
        optimizer, optimizer_adv,
        std_l1=0.5, mean_l1=0.01, weight_l1=0,
        backprop_sample_loss_to_cluster_weights=False):
    feature_model.train()
    all_trans_losses = []
    all_l1_losses = []
    all_losses = []
    for i_examples in get_balanced_batches(len(inputs), rng, shuffle=True,
                                           batch_size=batch_size):
        batch_X = inputs[th.LongTensor(i_examples)]
        trans_loss, threshold_l1_penalty, total_loss = train_on_batch(
            batch_X, feature_model,
            means_per_dim, stds_per_dim, weights_per_cluster,
            directions_adv,
            optimizer, optimizer_adv,
            std_l1=std_l1, mean_l1=mean_l1, weight_l1=weight_l1,
            backprop_sample_loss_to_cluster_weights=backprop_sample_loss_to_cluster_weights)
        all_trans_losses.append(var_to_np(trans_loss))
        all_l1_losses.append(var_to_np(threshold_l1_penalty))
        all_losses.append(var_to_np(total_loss))

    return all_trans_losses, all_l1_losses, all_losses


def train_on_batch(
        batch_X, feature_model,
        means_per_dim, stds_per_dim, weights_per_cluster,
        directions_adv,
            optimizer, optimizer_adv,
        std_l1, mean_l1, weight_l1,
        backprop_sample_loss_to_cluster_weights):
    batch_outs = feature_model(batch_X).squeeze()
    trans_losses = []
    for a_dir in [None, None, norm_and_var_directions(directions_adv)]:
        trans_loss = sample_transport_loss(
            batch_outs, means_per_dim, stds_per_dim,
            weights_per_cluster / th.sum(weights_per_cluster),
            'square', n_interpolation_samples=len(batch_outs) * 2,
            backprop_sample_loss_to_cluster_weights=backprop_sample_loss_to_cluster_weights,
            normalize_by_stds=False, energy_sample_loss=False,
            directions=a_dir)
        trans_losses.append(trans_loss)
    trans_loss = th.sum(th.cat(trans_losses))
    threshold_l1_penalty = ((th.mean(th.abs(stds_per_dim))) * std_l1 +
                            th.mean(th.abs(weights_per_cluster)) * weight_l1 +
                            th.mean(th.abs(means_per_dim)) * mean_l1)
    total_loss = trans_loss + threshold_l1_penalty
    optimizer.zero_grad()
    optimizer_adv.zero_grad()
    total_loss.backward()
    optimizer.step()
    # directions should try to increase loss.
    directions_adv.grad.data.neg_()
    optimizer_adv.step()
    weights_per_cluster.data.clamp_(min=0)
    weights_per_cluster.data.div_(th.sum(weights_per_cluster.data))
    stds_per_dim.data.clamp_(min=1e-4)
    return trans_loss, threshold_l1_penalty, total_loss


## Earlier stuff for Cramer distance/L2-distance of cumulative distribution functions

def analytical_l2_cdf_and_sample_transport_loss(
        samples, means_per_dim, stds_per_dim, weights_per_cluster, abs_or_square,
        n_interpolation_samples,
        cuda=False, directions=None, backprop_sample_loss_to_cluster_weights=True,
        normalize_by_stds=True, energy_sample_loss=False):
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
    if energy_sample_loss:
        sample_loss = sampled_energy_transport_loss(
            projected_samples, directions,
            means_per_dim, stds_per_dim, weights_per_cluster,
            abs_or_square,
            backprop_to_cluster_weights=backprop_sample_loss_to_cluster_weights,
            normalize_by_stds=normalize_by_stds)
    else:
        sample_loss =  sampled_transport_diffs_interpolate_sorted_part(
            sorted_samples, directions, means_per_dim,
            stds_per_dim, weights_per_cluster, n_interpolation_samples,
            abs_or_square=abs_or_square,
            backprop_to_cluster_weights=backprop_sample_loss_to_cluster_weights,
            normalize_by_stds=normalize_by_stds)
    return cdf_loss, sample_loss

def analytical_l2_cdf_loss(
        samples, means_per_dim, stds_per_dim, weights_per_cluster,
        cuda=False, directions=None, ):
    # common
    if directions is None:
        directions = sample_directions(samples.size()[1], True, cuda=cuda)
    else:
        directions = norm_and_var_directions(directions)

    projected_samples = th.mm(samples, directions.t())
    sorted_samples, _ = th.sort(projected_samples, dim=0)
    cdf_loss = analytical_l2_cdf_loss_given_sorted_samples(
        sorted_samples, directions,
        means_per_dim, stds_per_dim, weights_per_cluster)
    return cdf_loss

def analytical_l2_cdf_loss_given_sorted_samples(
        sorted_samples_batch, directions,
        means_per_dim, stds_per_dim, weights_per_cluster):
    n_samples = len(sorted_samples_batch)
    mean_dirs, std_dirs = transform_gaussian_by_dirs(
        means_per_dim, stds_per_dim, directions)
    assert (th.sum(weights_per_cluster) >= 0).data.all()
    normed_weights = weights_per_cluster / th.sum(weights_per_cluster)
    analytical_cdf = multi_directions_gaussian_cdfs(sorted_samples_batch.t(),
                                                    mean_dirs, std_dirs,
                                                    normed_weights)
    #empirical_cdf = th.linspace(1 / (n_samples + 1), 1 - (1 / (n_samples+1)),
    #                            n_samples).unsqueeze(0)
    empirical_cdf = th.linspace(1 / (n_samples), 1 - (1 / (n_samples)),
                                n_samples).unsqueeze(0)
    empirical_cdf = th.autograd.Variable(empirical_cdf)
    directions, empirical_cdf = ensure_on_same_device(directions, empirical_cdf)
    diffs = analytical_cdf - empirical_cdf
    cdf_loss = th.mean(th.sqrt(th.sum(diffs * diffs, dim=1)))
    #cdf_loss = th.mean(th.sqrt(th.mean(diffs * diffs, dim=1)))
    return cdf_loss




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

def multi_directions_gaussian_cdfs(x, means, stds, weights):
    # see https://stats.stackexchange.com/questions/187828/how-are-the-error-function-and-standard-normal-distribution-function-related
    # and https://en.wikipedia.org/wiki/Normal_distribution#Cumulative_distribution_function
    # assuming input x is directions x 1-dimensional points
    # assuming means/stds are directions x clusters
    # weights is 1-dimension (number of clusters)
    eps = 1e-6
    stds = th.clamp(stds, min=eps)
    weights = weights / th.sum(weights)
    cdfs = 0.5 * (1 +
                  th.erf((x.unsqueeze(2) - means.unsqueeze(1)) / (
                      stds.unsqueeze(1) * np.sqrt(2))))
    # directions x points x clusters
    cdf = th.sum(cdfs * weights.unsqueeze(0).unsqueeze(1), dim=2)
    # directions x points
    return cdf

