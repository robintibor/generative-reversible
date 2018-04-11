def semi_dual_transport_loss(outs, mean, std):
    gauss_samples = get_gauss_samples(len(outs), mean, std)
    return semi_dual_transport_loss_for_samples(outs, gauss_samples, v_network, u_network)


def semi_dual_transport_loss_for_samples(samples_a, samples_b, a_network, b_network):
    diffs = samples_a.unsqueeze(1) - samples_b.unsqueeze(0)
    diffs = th.sum((diffs * diffs), dim=2)
    samples_a = samples_a.detach()
    samples_b = samples_b.detach()
    cur_v = a_network(samples_a).squeeze(1)
    cur_u = b_network(samples_b).squeeze(1)
    diffs_v = diffs - cur_v.unsqueeze(1)
    diffs_u = diffs - cur_u.unsqueeze(0)
    min_diffs_v, _ = th.min(diffs_v, dim=0)
    min_diffs_u, _ = th.min(diffs_u, dim=1)
    ot_dist = th.mean(cur_v) + th.mean(cur_u)  + th.mean(min_diffs_u) + th.mean(min_diffs_v)
    return ot_dist
