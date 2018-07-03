import torch as th

def hard_init_std_mean(
        means_per_cluster, stds_per_cluster,
        feature_model, inputs, targets,):
    for i_cluster in range(len(means_per_cluster)):
        target_mask = targets[:, i_cluster] == 1
        for _ in range(1, len(inputs.size())):
            target_mask = target_mask.unsqueeze(1)
        this_ins = inputs[target_mask]
        new_shape = (-1,) + inputs.shape[1:]
        this_ins = this_ins.view(*new_shape)
        this_outs = feature_model(this_ins)
        mean = means_per_cluster[i_cluster]
        std = stds_per_cluster[i_cluster]
        emp_mean = th.mean(this_outs, dim=0)
        emp_std = th.std(this_outs, dim=0)
        mean.data = emp_mean.data
        std.data = emp_std.data