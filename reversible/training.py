from numpy.random import RandomState
import torch as th
import numpy as np

from reversible.ampphase import set_phase_interval_around_mean_in_outs
from reversible.util import var_to_np, enforce_2d
from reversible.sliced import sample_directions


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