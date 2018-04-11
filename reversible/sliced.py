import torch as th

from reversible.gaussian import get_gauss_samples


def sample_directions(n_dims, orthogonalize, cuda):
    if cuda:
        directions = th.cuda.FloatTensor(n_dims, n_dims).normal_(0, 1)
    else:
        directions = th.FloatTensor(n_dims, n_dims).normal_(0, 1)

    if orthogonalize:
        directions, _ = th.qr(directions)

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


def sliced_from_samples(samples_a, samples_b, n_dirs, adv_dirs):
    assert (n_dirs > 0) or (adv_dirs is not None)
    dirs = [sample_directions(samples_a.size()[1], orthogonalize=True,
                              cuda=samples_a.is_cuda) for _ in range(n_dirs)]
    if adv_dirs is not None:
        dirs = dirs + [adv_dirs]
    dirs = th.cat(dirs, dim=0)
    dirs = norm_and_var_directions(dirs)
    return sliced_from_samples_for_dirs(samples_a, samples_b, dirs)


def sliced_from_samples_for_dirs(samples_a, samples_b, dirs):
    projected_samples_a = th.mm(samples_a, dirs.t())
    sorted_samples_a, _ = th.sort(projected_samples_a, dim=0)
    projected_samples_b = th.mm(samples_b, dirs.t())
    sorted_samples_b, _ = th.sort(projected_samples_b, dim=0)
    diffs = sorted_samples_a - sorted_samples_b
    # first sum across examples
    # (one W2-value per direction)
    # then mean across directions
    # then sqrt
    loss = th.sqrt(th.mean(diffs * diffs))
    return loss


def sliced_from_samples_for_gauss_dist(outs, mean, std, n_dirs, adv_dirs):
    gauss_samples = get_gauss_samples(len(outs), mean, std)
    return sliced_from_samples(outs, gauss_samples, n_dirs, adv_dirs)

