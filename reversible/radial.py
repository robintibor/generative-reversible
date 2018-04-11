import torch as th

from reversible.gaussian import get_gauss_samples


def radial_distance_loss_from_samples_for_test_samples(samples_a, samples_b, test_samples,):
    diffs_a = samples_a.unsqueeze(0) - test_samples.unsqueeze(1)
    diffs_a = th.sqrt(th.sum((diffs_a * diffs_a), dim=2))
    sorted_diffs_a, _ = th.sort(diffs_a, dim=1)

    diffs_b = samples_b.unsqueeze(0) - test_samples.unsqueeze(1)
    diffs_b = th.sqrt(th.sum((diffs_b * diffs_b), dim=2))
    sorted_diffs_b, _ = th.sort(diffs_b, dim=1)

    diff_diff = (sorted_diffs_a - sorted_diffs_b)
    loss = th.sqrt(th.mean(diff_diff * diff_diff))
    return loss


def radial_distance_loss_from_samples_for_gauss_dist(outs, mean, std, n_test_samples, adv_samples):
    if (n_test_samples is not None) and n_test_samples > 0:
        test_samples = get_gauss_samples(n_test_samples, mean, std)
    else:
        test_samples = None
    if adv_samples is not None:
        if test_samples is not None:
            test_samples = th.cat((test_samples, adv_samples), dim=0)
        else:
            test_samples = adv_samples
    gauss_samples = get_gauss_samples(len(outs), mean, std)
    return radial_distance_loss_from_samples_for_test_samples(outs, gauss_samples, test_samples)
