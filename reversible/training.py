from numpy.random import RandomState
import torch as th
import numpy as np
from reversible.ampphase import set_phase_interval_around_mean_in_outs
from reversible.util import var_to_np, enforce_2d, np_to_var
from reversible.sliced import sample_directions
from reversible.util import ensure_on_same_device


def compute_embedding_loss(diffs, i_example_to_i_samples):
    main_loss = 0
    for i_example, i_samples in enumerate(i_example_to_i_samples):
        for i_sample in i_samples:
            main_loss += diffs[i_example, i_sample]
    main_loss = th.sqrt(main_loss / diffs.size()[0] + 1e-6)
    return main_loss


class GenerativeRevTrainer(object):
    def __init__(self, model, optimizer, means_per_dim, stds_per_dim,
                 iterator):
        self.__dict__.update(locals())
        del self.self
        self.rng = RandomState(394834)

    def train_epoch(self, inputs, targets, inputs_u, linear_weights_u,
                    trans_loss_function,
                    directions_adv, n_dir_matrices=1):
        loss = 0
        n_examples = 0
        for batch_X, batch_y in self.iterator.get_batches(
                inputs, targets, inputs_u, linear_weights_u):
            if n_dir_matrices > 0:
                dir_mats = [sample_directions(self.means_per_dim.size()[1], True,
                                              cuda=batch_X.is_cuda)
                            for _ in range(n_dir_matrices)]
                directions = th.cat(dir_mats, dim=0)
                if directions_adv is not None:
                    directions = th.cat((directions, directions_adv), dim=0)
            else:
                directions = directions_adv
            batch_loss = train_on_batch(batch_X, self.model, self.means_per_dim,
                                        self.stds_per_dim,
                                        batch_y, self.optimizer, directions,
                                        trans_loss_function)
            loss = loss + batch_loss * len(batch_X)
            n_examples = n_examples + batch_X.size()[0]
        mean_loss = var_to_np(loss / n_examples)[0]
        return mean_loss


def train_on_batch(batch_X, model, means_per_dim, stds_per_dim, soft_targets,
                   optimizer,
                   directions, trans_loss_function,
                   ):
    if means_per_dim.is_cuda:
        batch_X = batch_X.cuda()
        soft_targets = soft_targets.cuda()

    batch_outs = model(batch_X)
    batch_outs = enforce_2d(batch_outs)
    trans_loss = trans_loss_function(batch_outs, directions, soft_targets,
                                     means_per_dim, stds_per_dim)
    optimizer.zero_grad()
    trans_loss.backward()
    optimizer.step()
    return trans_loss


def init_std_mean(feature_model, inputs, targets, means_per_dim, stds_per_dim,
                  set_phase_interval):
    outs = feature_model(inputs)
    outs = enforce_2d(outs)
    for i_cluster in range(len(means_per_dim)):
        this_weights = targets[:, i_cluster]
        n_elems = len(th.nonzero(this_weights == 1))
        this_outs = outs
        if set_phase_interval:
            this_outs = set_phase_interval_around_mean_in_outs(
                this_outs, this_weights=this_weights)
        this_outs = this_outs[(this_weights == 1).unsqueeze(1)].resize(
            n_elems, this_outs.size()[1])

        means = th.mean(this_outs, dim=0)
        # this_outs = uniform_to_gaussian_phases_in_outs(this_outs,means)
        stds = th.std(this_outs, dim=0)
        means_per_dim.data[i_cluster] = means.data
        stds_per_dim.data[i_cluster] = stds.data


def select_outs_from_targets(outs, targets, i_cluster):
    return outs[(targets[:, i_cluster] == 1).unsqueeze(1)].view(
            -1, outs.size()[1])

def select_from_targets(a, targets, i_cluster):
    mask = targets[:, i_cluster] == 1
    new_shape = (-1,) + a.size()[1:]
    for _ in range(1, len(a.size())):
        mask = mask.unsqueeze(1)
    return a[mask].view(new_shape)


def hard_init_std_mean(feature_model, inputs, targets, means_per_cluster,
                       stds_per_cluster, ):
    for i_cluster in range(len(means_per_cluster)):
        target_mask = targets[:, i_cluster] == 1
        for _ in range(1, len(inputs.size())):
            target_mask = target_mask.unsqueeze(1)
        this_ins = inputs[target_mask]
        new_shape = (-1,) + inputs.shape[1:]
        this_ins = this_ins.view(*new_shape)
        this_outs = feature_model(this_ins)
        mean = means_per_cluster[i_cluster]
        std = stds_per_cluster[i_cluster]
        emp_mean = th.mean(this_outs, dim=0)
        emp_std = th.std(this_outs, dim=0)
        mean.data = emp_mean.data
        std.data = emp_std.data

def get_batch(inputs, targets, rng, batch_size, with_replacement, i_class='all', ):
    if i_class == 'all':
        indices = list(range(len(inputs)))
    else:
        indices = np.flatnonzero(var_to_np(targets[:,i_class]) == 1)
    batch_inds = rng.choice(indices, size=batch_size, replace=with_replacement)
    th_inds = np_to_var(batch_inds, dtype=np.int64)
    th_inds, _ = ensure_on_same_device(th_inds, inputs)
    batch_X = inputs[th_inds]
    batch_y = targets[th_inds]
    return th_inds, batch_X, batch_y
