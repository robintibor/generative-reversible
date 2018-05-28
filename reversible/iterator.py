import numpy as np
from numpy.random import RandomState
import torch as th

from reversible.util import np_to_var


class GenerativeIterator(object):
    def __init__(self, upsample_supervised, batch_size):
        self.upsample_supervised = upsample_supervised
        self.rng = RandomState(3847634)
        self.batch_size = batch_size

    def get_batches(self, inputs, targets, inputs_u, linear_weights_u):
        i_supervised = np.arange(len(inputs))
        if inputs_u is not None:
            i_unsupervised = np.arange(len(inputs_u))
        if (inputs_u is not None) and self.upsample_supervised:
            assert inputs_u is not None
            n_missing = len(i_unsupervised) - len(i_supervised)
            while n_missing > 0:
                i_add = self.rng.choice(np.arange(len(inputs)), size=n_missing,
                                        replace=False)
                i_supervised = np.concatenate((i_supervised, i_add))
        longtype = th.LongTensor
        if inputs.is_cuda:
            longtype = th.cuda.LongTensor
        if inputs_u is not None:
            supervised_batch_size = int(np.round(self.batch_size *
                                                 (len(i_supervised) / float(
                                                     len(i_supervised) + len(
                                                         i_unsupervised)))))
            unsupervised_batch_size = self.batch_size - supervised_batch_size
            batches_supervised = get_exact_size_batches(
                len(i_supervised), self.rng, supervised_batch_size)
            batches_unsupervised = get_exact_size_batches(
                len(i_unsupervised), self.rng, unsupervised_batch_size)
            assert len(batches_supervised) == len(batches_unsupervised)
            # else you should check which one is smaller and add the first batch as the last batch again
            # or some random examples as another batch
            for batch_i_s, batch_i_u in zip(batches_supervised,
                                            batches_unsupervised):
                batch_X_s = inputs[longtype(batch_i_s)]
                batch_X_u = inputs_u[longtype(batch_i_u)]
                batch_y_s = targets[longtype(batch_i_s)]
                batch_y_u = linear_weights_u[longtype(batch_i_u)]
                batch_y_u = th.nn.functional.softmax(batch_y_u, dim=1)
                batch_X = th.cat((batch_X_s, batch_X_u), dim=0)
                batch_y = th.cat((batch_y_s, batch_y_u), dim=0)
                yield batch_X, batch_y
        else:
            supervised_batch_size = self.batch_size
            batches_supervised = get_exact_size_batches(
                len(i_supervised), self.rng, supervised_batch_size)
            for batch_i_s in batches_supervised:
                batch_X_s = inputs[longtype(batch_i_s)]
                batch_y_s = targets[longtype(batch_i_s)]
                yield batch_X_s, batch_y_s


def get_exact_size_batches(n_trials, rng, batch_size):
    i_trials = np.arange(n_trials)
    rng.shuffle(i_trials)
    i_trial = 0
    batches = []
    for i_trial in range(0, n_trials - batch_size, batch_size):
        batches.append(i_trials[i_trial: i_trial + batch_size])
    i_trial = i_trial + batch_size

    last_batch = i_trials[i_trial:]
    n_remain = batch_size - len(last_batch)
    last_batch = np.concatenate((last_batch, i_trials[:n_remain]))
    batches.append(last_batch)
    return batches


def get_batches_equal_classes(targets, n_classes, rng, batch_size):
    batches_per_cluster = []
    for i_cluster in range(n_classes):
        n_examples = np.sum(targets == i_cluster)
        examples_per_batch = get_exact_size_batches(
            n_examples, rng, batch_size)
        this_cluster_indices = np.nonzero(targets == i_cluster)[0]
        examples_per_batch = [this_cluster_indices[b] for b in
                              examples_per_batch]
        # revert back to actual indices
        batches_per_cluster.append(examples_per_batch)

    # hm is this correctsense? Intended is to transform
    # 3dimensional tensor: #classes x #batches x #examples
    # into 2dimensional tensor #batches x #examples
    # where each batch i_b is made by concatenating the batches i_b_c of each class
    batches = np.concatenate(batches_per_cluster, axis=1)
    return batches


def get_balanced_batches(n_trials, rng, shuffle, n_batches=None,
                         batch_size=None):
    """Create indices for batches balanced in size
    (batches will have maximum size difference of 1).
    Supply either batch size or number of batches. Resulting batches
    will not have the given batch size but rather the next largest batch size
    that allows to split the set into balanced batches (maximum size difference 1).

    Parameters
    ----------
    n_trials : int
        Size of set.
    rng : RandomState

    shuffle : bool
        Whether to shuffle indices before splitting set.
    n_batches : int, optional
    batch_size : int, optional

    Returns
    -------

    """
    assert batch_size is not None or n_batches is not None
    if n_batches is None:
        n_batches = int(np.round(n_trials / float(batch_size)))

    if n_batches > 0:
        min_batch_size = n_trials // n_batches
        n_batches_with_extra_trial = n_trials % n_batches
    else:
        n_batches = 1
        min_batch_size = n_trials
        n_batches_with_extra_trial = 0
    assert n_batches_with_extra_trial < n_batches
    all_inds = np.array(range(n_trials))
    if shuffle:
        rng.shuffle(all_inds)
    i_start_trial = 0
    i_stop_trial = 0
    batches = []
    for i_batch in range(n_batches):
        i_stop_trial += min_batch_size
        if i_batch < n_batches_with_extra_trial:
            i_stop_trial += 1
        batch_inds = all_inds[range(i_start_trial, i_stop_trial)]
        batches.append(batch_inds)
        i_start_trial = i_stop_trial
    assert i_start_trial == n_trials
    return batches


class BalancedBatchSizeIterator(object):
    """
    Create batches of balanced size.

    Parameters
    ----------
    batch_size: int
        Resulting batches will not necessarily have the given batch size
        but rather the next largest batch size that allows to split the set into
        balanced batches (maximum size difference 1).
    """

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.rng = RandomState(328774)

    def get_batches(self, X, y, shuffle):
        n_trials = len(X)
        batches = get_balanced_batches(n_trials,
                                       batch_size=self.batch_size,
                                       rng=self.rng,
                                       shuffle=shuffle)
        for batch_inds in batches:
            batch_inds = np_to_var(batch_inds, dtype=np.int64)
            if X.is_cuda:
                batch_inds = batch_inds.cuda()
            batch_X = X[batch_inds]
            batch_y = y[batch_inds]

            yield (batch_X, batch_y)

    def reset_rng(self):
        self.rng = RandomState(328774)
