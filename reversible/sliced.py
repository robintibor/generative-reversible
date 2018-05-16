import torch as th

from reversible.gaussian import get_gauss_samples


def sample_directions(n_dims, orthogonalize, cuda):
    if cuda:
        directions = th.cuda.FloatTensor(n_dims, n_dims).normal_(0, 1) + 1e-6
    else:
        directions = th.FloatTensor(n_dims, n_dims).normal_(0, 1) + 1e-6

    if orthogonalize:
        directions, r = th.qr(directions)
        d = th.diag(r, 0)
        ph = d.sign()
        directions *= ph
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


def sliced_from_samples(samples_a, samples_b, n_dirs, adv_dirs,
                        orthogonalize=True, dist='w2'):
    assert (n_dirs > 0) or (adv_dirs is not None)
    dirs = [sample_directions(samples_a.size()[1], orthogonalize=orthogonalize,
                              cuda=samples_a.is_cuda) for _ in range(n_dirs)]
    if adv_dirs is not None:
        dirs = dirs + [adv_dirs]
    dirs = th.cat(dirs, dim=0)
    dirs = norm_and_var_directions(dirs)
    return sliced_from_samples_for_dirs(samples_a, samples_b, dirs, dist=dist)


def sliced_from_samples_for_dirs(samples_a, samples_b, dirs, dist):
    assert dist in ['w2', 'sqw2']
    projected_samples_a = th.mm(samples_a, dirs.t())
    sorted_samples_a, _ = th.sort(projected_samples_a, dim=0)
    projected_samples_b = th.mm(samples_b, dirs.t())
    sorted_samples_b, _ = th.sort(projected_samples_b, dim=0)
    n_a = len(sorted_samples_a)
    n_b = len(sorted_samples_b)
    if n_a > n_b:
        assert n_a % n_b == 0
        increase_factor = n_a // n_b
        sorted_samples_a = sorted_samples_a.view(n_a // increase_factor,
                                                 increase_factor,
                                                 sorted_samples_a.size()[1])
        sorted_samples_b = sorted_samples_b.unsqueeze(1)
    elif n_a < n_b:
        assert n_b % n_a == 0
        increase_factor = n_b // n_a
        sorted_samples_b = sorted_samples_b.view(n_b // increase_factor,
                                                 increase_factor,
                                                 sorted_samples_b.size()[1])
        sorted_samples_a = sorted_samples_a.unsqueeze(1)

    diffs = sorted_samples_a - sorted_samples_b
    # first sum across examples
    # (one W2-value per direction)
    # then mean across directions
    # then sqrt
    if dist == 'w2':
        eps = 1e-6
        loss = th.sqrt(th.mean(diffs * diffs) + eps)
    else:
        assert dist == 'sqw2'
        loss = th.mean(diffs * diffs)
    return loss


def sliced_from_samples_for_gauss_dist(outs, mean, std, n_dirs, adv_dirs, **kwargs):
    gauss_samples = get_gauss_samples(len(outs), mean, std)
    return sliced_from_samples(outs, gauss_samples, n_dirs, adv_dirs, **kwargs)


def sliced_loss_for_dirs_3d(samples_full_a, samples_full_b, directions, dist):
    assert dist in ['w2', 'sqw2']
    proj_a = th.matmul(samples_full_a, directions.t())
    proj_b = th.matmul(samples_full_b, directions.t())
    sorted_a, _ = th.sort(proj_a, dim=1)
    sorted_b, _ = th.sort(proj_b, dim=1)
    # sorted are examples x locations x dirs
    # 
    #euclid_loss = th.mean(th.mean(th.sqrt(eps + th.mean((sorted_a - sorted_b) ** 2, dim=2)), dim=1), dim=0)
    if dist == 'w2':
        eps = 1e-6
        loss = th.sqrt(th.mean((sorted_a - sorted_b) ** 2) + eps)
    else:
        assert dist == 'sqw2'
        loss = th.mean((sorted_a - sorted_b) ** 2)
    return loss


def layer_sliced_loss(this_all_outs, wanted_all_outs, dist='w2'):
    layer_losses = []
    # could also think to exclude second to last layer as well
    # as it is just last viewed in different shape
    for i_layer in range(len(this_all_outs) - 1): # final layer is already matched and no transport makes sense there...
        layer_outs = this_all_outs[i_layer]
        layer_wanted_outs = wanted_all_outs[i_layer]
        directions = sample_directions(n_dims=layer_outs.size()[1], orthogonalize=True, cuda=True)
        samples_full_a = layer_outs.contiguous().view(layer_outs.size()[0], layer_outs.size()[1], -1).permute(0,2,1)
        samples_full_b = layer_wanted_outs.contiguous().view(
            layer_wanted_outs.size()[0],
            layer_wanted_outs.size()[1],
            -1).permute(0,2,1)
        layer_loss = sliced_loss_for_dirs_3d(samples_full_a, samples_full_b, directions,
                                             dist=dist)
        layer_losses.append(layer_loss)
    total_loss = th.mean(th.cat(layer_losses, dim=0))
    return total_los

