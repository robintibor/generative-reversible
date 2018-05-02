def hard_loss_per_cluster(outs, targets, means_per_dim, stds_per_dim,
                          hard_loss_fn):
    """Assume each output belongs only to one cluster (hard assignment),
    and correspondingly compute losses per cluster individually."""
    loss = 0
    for i_cluster in range(len(means_per_dim)):
        mean = means_per_dim[i_cluster]
        std = stds_per_dim[i_cluster]
        this_outs = outs[(targets[:, i_cluster] == 1).unsqueeze(1)].view(
            -1, outs.size()[1])
        this_loss = hard_loss_fn(this_outs, mean, std)
        loss = this_loss + loss
    return loss / float(len(means_per_dim))


def pairwise_squared_dist_2d(a,b):
    diffs = a.unsqueeze(1) - b.unsqueeze(0)
    return th.sum(diffs * diffs, dim=2)


def squared_dist_2d_matched(a,b):
    diffs = a - b
    return th.sum(diffs * diffs, dim=1)

