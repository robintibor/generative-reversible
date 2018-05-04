import torch as th
from reversible.sliced import sample_directions, norm_and_var_directions


def standardize_by_outs(this_all_outs, wanted_all_outs):
    standardized_l_out = []
    standardized_l_wanted = []
    for i_layer in range(len(this_all_outs)):
        l_out = this_all_outs[i_layer]
        l_reshaped = l_out.transpose(1,0).contiguous().view(l_out.size()[1], -1)
        means = th.mean(l_reshaped, dim=1)
        stds = th.std(l_reshaped, dim=1)
        mean_std = th.mean(stds)
        l_standardized = (l_reshaped - means.unsqueeze(1)) / mean_std
        l_standardized = l_standardized.transpose(1,0).contiguous().view(*l_out.size())
        standardized_l_out.append(l_standardized)
        l_wanted_out = wanted_all_outs[i_layer]
        l_wanted_reshaped = l_wanted_out.transpose(1,0).contiguous().view(l_wanted_out.size()[1], -1)
        l_wanted_standardized = (l_wanted_reshaped - means.unsqueeze(1)) / mean_std
        l_wanted_standardized = l_wanted_standardized.contiguous().view(*l_wanted_out.size())
        standardized_l_wanted.append(l_wanted_standardized)
    return standardized_l_out, standardized_l_wanted


def sliced_loss_for_dirs_3d(samples_full_a, samples_full_b, directions, diff_fn):
    proj_a = th.matmul(samples_full_a, directions.t())
    proj_b = th.matmul(samples_full_b, directions.t())
    sorted_a, _ = th.sort(proj_a, dim=1)
    sorted_b, _ = th.sort(proj_b, dim=1)
    # sorted are examples x locations x dirs

    #eps = 1e-6
    #euclid_loss = th.mean(th.mean(th.sqrt(eps + th.mean((sorted_a - sorted_b) ** 2, dim=2)), dim=1), dim=0)
    if diff_fn == 'w2':
        eps = 1e-6
        loss = th.sqrt(th.mean(th.mean(th.mean((sorted_a - sorted_b) ** 2, dim=2), dim=1), dim=0)+ eps)
    elif diff_fn == 'sqw2':
        loss = th.mean(th.mean(th.mean((sorted_a - sorted_b) ** 2, dim=2), dim=1), dim=0)
    elif diff_fn == 'fakeeuclid':
        eps = 1e-6
        loss = th.mean(th.mean(th.sqrt(th.sum((sorted_a - sorted_b) ** 2, dim=2)+ eps), dim=1), dim=0)
    return loss


def layer_sliced_loss(this_all_outs, wanted_all_outs, return_all=False, orthogonalize=True,
                    adv_dirs=None,
                      diff_fn='w2'):
    layer_losses = []
    for i_layer in range(len(this_all_outs)):
        layer_outs = this_all_outs[i_layer]
        layer_wanted_outs = wanted_all_outs[i_layer]
        directions = sample_directions(n_dims=layer_outs.size()[1], orthogonalize=orthogonalize, cuda=True)
        if adv_dirs is not None:
            this_adv_dirs = norm_and_var_directions(adv_dirs[i_layer])
            directions = th.cat((directions, this_adv_dirs), dim=0)
        samples_full_a = layer_outs.contiguous().view(layer_outs.size()[0], layer_outs.size()[1], -1).permute(0,2,1)
        samples_full_b = layer_wanted_outs.contiguous().view(
            layer_wanted_outs.size()[0],
            layer_wanted_outs.size()[1],
            -1).permute(0,2,1)
        layer_loss = sliced_loss_for_dirs_3d(samples_full_a, samples_full_b, directions, diff_fn=diff_fn)
        layer_losses.append(layer_loss)
    if return_all:
        return th.cat(layer_losses, dim=0)
    else:
        total_loss = th.mean(th.cat(layer_losses, dim=0))
        return total_loss
