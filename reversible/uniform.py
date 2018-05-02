import torch as th

def get_uniform_samples(n_samples, mean, spread):
    if mean.is_cuda:
        orig_samples = th.cuda.FloatTensor(n_samples, len(mean)).uniform_(0, 1)
    else:
        orig_samples = th.FloatTensor(n_samples, len(mean)).uniform_(0, 1)
    orig_samples = th.autograd.Variable(orig_samples)
    samples = (orig_samples * spread.unsqueeze(0)) + mean.unsqueeze(0)
    return samples