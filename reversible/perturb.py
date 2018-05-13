import numpy as np
import torch as th
from torch import nn

from reversible.gaussian import get_gauss_samples
from reversible.revnet import invert
from reversible.uniform import get_uniform_samples
from reversible.util import ensure_on_same_device


def invert_with_perturb(feature_model, outs, i_layers_to_perturb, perturb_norm=0.01,
                        perturb_dist='uniform'):
    l_in = outs
    l_ins = []

    enumerated_modules = list(enumerate(list(feature_model.children())))
    for i_module, module in enumerated_modules[::-1]:
        l_ins.append(l_in)
        l_in = invert(nn.Sequential(module),l_in)
        if i_module in i_layers_to_perturb:
            perturbations = get_perturbations(l_in, norm=perturb_norm,
                                              perturb_dist=perturb_dist)
            l_in = l_in + perturbations

    l_ins.append(l_in)
    return l_ins


def get_perturbed_outs(feature_model, ins, i_layers_to_perturb,
                       perturb_norm=0.01, perturb_dist='uniform'):
    l_out = ins
    l_outs = []
    for i_module, module in enumerate(feature_model.children()):
        l_outs.append(l_out)
        if i_module in i_layers_to_perturb:  # perturb input to layers to perturb
            perturbations = get_perturbations(l_out, norm=perturb_norm,
                                              perturb_dist=perturb_dist)
            l_out = l_out + perturbations
        l_out = module(l_out)
    if (i_module + 1) in i_layers_to_perturb:
        perturbations = get_perturbations(l_out, norm=perturb_norm,
                                          perturb_dist=perturb_dist)
        l_out = l_out + perturbations

    l_outs.append(l_out)
    return l_outs

def get_perturbations(l_out, norm, perturb_dist):
    dims = int(np.prod(l_out.size()[1:]))
    mean = th.zeros(dims)
    std = th.ones(dims)
    mean = th.autograd.Variable(mean)
    std = th.autograd.Variable(std)
    _, mean, std = ensure_on_same_device(l_out, mean, std)
    if perturb_dist == 'uniform':
        perturbations = get_uniform_samples(l_out.size()[0], mean, std)
    else:
        assert perturb_dist == 'gaussian'
        perturbations = get_gauss_samples(l_out.size()[0], mean, std)

    perturbations = norm * (perturbations / th.sqrt(th.sum(perturbations ** 2, dim=1, keepdim=True)))
    perturbations = perturbations.view(l_out.size())
    return perturbations