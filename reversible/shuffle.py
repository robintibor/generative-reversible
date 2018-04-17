import torch as th

from reversible.util import ensure_on_same_device


def get_features_shuffled(x):
    randperms = th.autograd.Variable(
        th.stack([th.randperm(len(x)) for _ in range(x.shape[1])], dim=1))
    randperms, x = ensure_on_same_device(randperms, x)
    shuffled_ins = x.gather(index=randperms, dim=0)
    return shuffled_ins
