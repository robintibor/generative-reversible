import torch as th

from reversible.gaussian import get_gauss_samples
from reversible.util import log_sum_exp, ensure_on_same_device, var_to_np


def sinkhorn_to_gauss_dist(outs, mean, std, **kwargs):
    gauss_samples = get_gauss_samples(len(outs), mean, std)
    return sinkhorn_sample_loss(outs, gauss_samples, **kwargs)


def M(u, v, C, epsilon):
    "Modified cost for logarithmic updates"
    "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
    return (-C + u.unsqueeze(1) + v.unsqueeze(0)) / epsilon


def sinkhorn_sample_loss(samples_a, samples_b, epsilon=0.01, stop_threshold=0.1,
                         max_iters=50, normalize_cost_matrix=False, max_normed_entropy=None):
    assert normalize_cost_matrix in [False, 'mean', 'max']
    diffs = samples_a.unsqueeze(1) - samples_b.unsqueeze(0)
    C = th.sum(diffs * diffs, dim=2)
    del diffs
    C_nograd = C.detach()
    if normalize_cost_matrix == 'mean':
        C_nograd = C_nograd / th.mean(C_nograd)
    elif normalize_cost_matrix == 'max':
        C_nograd = C_nograd / th.max(C_nograd)

    if max_normed_entropy is None:
        estimated_trans_th = estimate_transport_matrix_sinkhorn(
            C_nograd, epsilon=epsilon, stop_threshold=stop_threshold,
            max_iters=max_iters)
    else:
        estimated_trans_th, _ = transport_mat_sinkhorn_below_entropy(
            C_nograd, start_eps=epsilon, stop_threshold=stop_threshold,
            max_iters_sinkhorn=max_iters, max_iters_for_entropy=10,
            max_normed_entropy=max_normed_entropy)

    cost = th.sqrt(th.sum(estimated_trans_th * C))  # Sinkhorn cost
    return cost


def transport_mat_sinkhorn_below_entropy(
        C, start_eps, max_normed_entropy, max_iters_for_entropy,
        max_iters_sinkhorn=50, stop_threshold=1e-3):
    normed_entropy = max_normed_entropy + 1
    iteration = 0
    cur_eps = start_eps
    while (normed_entropy > max_normed_entropy) and (iteration < max_iters_for_entropy):

        transport_mat = estimate_transport_matrix_sinkhorn(
            C, epsilon=cur_eps, stop_threshold=stop_threshold, max_iters=max_iters_sinkhorn)
        relevant_mat = transport_mat[transport_mat > 0]
        normed_entropy = -th.sum(relevant_mat * th.log(relevant_mat)) / np.log(transport_mat.numel() * 1.)
        normed_entropy = var_to_np(normed_entropy)
        iteration += 1
        cur_eps = cur_eps / 2

    return transport_mat, cur_eps

def estimate_transport_matrix_sinkhorn(C, epsilon=0.01, stop_threshold=0.1,
                                       max_iters=50):
    n1 = C.size()[0]
    n2 = C.size()[1]
    mu = th.autograd.Variable(1. / n1 * th.FloatTensor(n1).fill_(1),
                              requires_grad=False)
    nu = th.autograd.Variable(1. / n2 * th.FloatTensor(n2).fill_(1),
                              requires_grad=False)
    mu, nu, C = ensure_on_same_device(mu, nu, C)
    u, v, err = 0. * mu, 0. * nu, 0.
    actual_nits = 0  # to check if algorithm terminates because of threshold or max iterations reached
    for i in range(max_iters):
        u1 = u  # useful to check the update
        u = epsilon * (
            th.log(mu) - log_sum_exp(M(u, v, C, epsilon), dim=1, keepdim=True).squeeze()) + u
        v = epsilon * (
            th.log(nu) - log_sum_exp(M(u, v, C, epsilon).t(), dim=1, keepdim=True).squeeze()) + v
        err = (u - u1).abs().sum()

        actual_nits += 1
        if var_to_np(err < stop_threshold).all():
            break
    estimated_transport_matrix = th.exp(M(u, v, C, epsilon))
    return estimated_transport_matrix
