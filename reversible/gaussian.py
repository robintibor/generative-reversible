import torch as th
import numpy as np

# in case you want truncated, see
#https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
# torch.fmod(torch.randn(size),2) (or torch.fmod stdf aft
def get_gauss_samples(n_samples, mean, std, truncate=False):
    if mean.is_cuda:
        orig_samples = th.cuda.FloatTensor(n_samples, len(mean)).normal_(0, 1)
    else:
        orig_samples = th.FloatTensor(n_samples, len(mean)).normal_(0, 1)
    if truncate:
        orig_samples = th.fmod(orig_samples, 3)
    orig_samples = th.autograd.Variable(orig_samples)
    samples = (orig_samples * std.unsqueeze(0)) + mean.unsqueeze(0)
    return samples

