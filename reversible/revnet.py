### Reversible model parts

import torch as th

from reversible.ampphase import amp_phase_to_x_y
from reversible.ampphase import gaussian_to_uniform_phases_in_outs
from reversible.gaussian import sizes_from_weights, sample_mixture_gaussian
from reversible.util import var_to_np


class ReversibleBlock(th.nn.Module):
    def __init__(self, F, G):
        super(ReversibleBlock, self).__init__()
        self.F = F
        self.G = G

    def forward(self, x):
        n_chans = x.size()[1]
        assert n_chans % 2 == 0
        x1 = x[:, :n_chans // 2]
        x2 = x[:, n_chans // 2:]
        y1 = self.F(x1) + x2
        y2 = self.G(y1) + x1
        return th.cat((y1, y2), dim=1)


class SubsampleSplitter(th.nn.Module):
    def __init__(self, stride, chunk_chans_first=True, checkerboard=False):
        super(SubsampleSplitter, self).__init__()
        if not hasattr(stride, '__len__'):
            stride = (stride, stride)
        self.stride = stride
        self.chunk_chans_first = chunk_chans_first
        self.checkerboard = checkerboard
        if checkerboard:
            assert stride[0] == 2
            assert stride[1] == 2

    def forward(self, x):
        # Chunk chans first to ensure that each of the two streams in the
        # reversible network will see a subsampled version of the whole input
        # (in case the preceding blocks would not alter the input)
        # and not one half of the input
        new_x = []
        if self.chunk_chans_first:
            xs = th.chunk(x, 2, dim=1)
        else:
            xs = [x]
        for one_x in xs:
            if not self.checkerboard:
                for i_stride in range(self.stride[0]):
                    for j_stride in range(self.stride[1]):
                        new_x.append(
                            one_x[:, :, i_stride::self.stride[0],
                            j_stride::self.stride[1]])
            else:
                new_x.append(one_x[:,:,0::2,0::2])
                new_x.append(one_x[:,:,1::2,1::2])
                new_x.append(one_x[:,:,0::2,1::2])
                new_x.append(one_x[:,:,1::2,0::2])

        new_x = th.cat(new_x, dim=1)
        return new_x


class ViewAs(th.nn.Module):
    def __init__(self, dims_before, dims_after):
        super(ViewAs, self).__init__()
        self.dims_before = dims_before
        self.dims_after = dims_after

    def forward(self, x):
        for i_dim in range(len(x.size())):
            expected = self.dims_before[i_dim]
            if expected != -1:
                assert x.size()[i_dim] == expected
        return x.view(self.dims_after)


def invert(feature_model, features):
    if feature_model.__class__.__name__ == 'ReversibleBlock' or feature_model.__class__.__name__ == 'SubsampleSplitter':
        feature_model = th.nn.Sequential(feature_model, )
    for module in reversed(list(feature_model.children())):
        if module.__class__.__name__ == 'ReversibleBlock':
            n_chans = features.size()[1]
            # y1 = self.F(x1) + x2
            # y2 = self.G(y1) + x1
            y1 = features[:, :n_chans // 2]
            y2 = features[:, n_chans // 2:]

            x1 = y2 - module.G(y1)
            x2 = y1 - module.F(x1)
            features = th.cat((x1, x2), dim=1)
        if module.__class__.__name__ == 'ReversibleBlockMemCNN':
            n_chans = features.size()[1]
            y1 = features[:, :n_chans // 2]
            y2 = features[:, n_chans // 2:]
            #y1 = F(x2) + x1
            #y2 = G(y1) + x2
            x2 = y2 - module.Gm(y1)
            x1 = y1 - module.Fm(x2)
            features = th.cat((x1, x2), dim=1)
        if module.__class__.__name__ == 'SubsampleSplitter':
            # after splitting the input into two along channel dimension if possible
            # for i_stride in range(self.stride):
            #    for j_stride in range(self.stride):
            #        new_x.append(one_x[:,:,i_stride::self.stride, j_stride::self.stride])
            n_all_chans_before = features.size()[1] // (
            module.stride[0] * module.stride[1])
            # if ther was only one chan before, chunk had no effect
            if module.chunk_chans_first and (n_all_chans_before > 1):
                chan_features = th.chunk(features, 2, dim=1)
            else:
                chan_features = [features]
            all_previous_features = []
            for one_chan_features in chan_features:
                previous_features = th.zeros(one_chan_features.size()[0],
                                             one_chan_features.size()[1] // (
                                             module.stride[0] * module.stride[
                                                 1]),
                                             one_chan_features.size()[2] *
                                             module.stride[0],
                                             one_chan_features.size()[3] *
                                             module.stride[1])
                if features.is_cuda:
                    previous_features = previous_features.cuda()
                previous_features = th.autograd.Variable(previous_features)

                n_chans_before = previous_features.size()[1]
                cur_chan = 0
                if not module.checkerboard:
                    for i_stride in range(module.stride[0]):
                        for j_stride in range(module.stride[1]):
                            previous_features[:, :, i_stride::module.stride[0],
                                    j_stride::module.stride[1]] = (
                                one_chan_features[:,
                                cur_chan * n_chans_before:cur_chan * n_chans_before + n_chans_before])
                            cur_chan += 1
                else:
                    # Manually go through 4 checkerboard positions
                    assert module.stride[0] == 2
                    assert module.stride[1] == 2
                    previous_features[:, :, 0::2,0::2] = (
                        one_chan_features[:,
                        0 * n_chans_before:0 * n_chans_before + n_chans_before])
                    previous_features[:, :, 1::2,1::2] = (
                        one_chan_features[:,
                        1 * n_chans_before:1 * n_chans_before + n_chans_before])
                    previous_features[:, :, 0::2,1::2] = (
                        one_chan_features[:,
                        2 * n_chans_before:2 * n_chans_before + n_chans_before])
                    previous_features[:, :, 1::2,0::2] = (
                        one_chan_features[:,
                        3 * n_chans_before:3 * n_chans_before + n_chans_before])
                all_previous_features.append(previous_features)
            features = th.cat(all_previous_features, dim=1)
        if module.__class__.__name__ == 'AmplitudePhase':
            n_chans = features.size()[1]
            assert n_chans % 2 == 0
            amps = features[:, :n_chans // 2]
            phases = features[:, n_chans // 2:]
            x1, x2 = amp_phase_to_x_y(amps, phases)
            features = th.cat((x1, x2), dim=1)
        if module.__class__.__name__ == 'ViewAs':
            for i_dim in range(len(features.size())):
                expected = module.dims_after[i_dim]
                if expected != -1:
                    assert features.size()[i_dim] == expected
            features = features.view(module.dims_before)
        if module.__class__.__name__ == 'ConstantPad2d':
            assert len(module.padding) == 4
            left, right, top, bottom = module.padding  # see pytorch docs
            features = features[:, :, top:-bottom, left:-right]  # see pytorch docs
    return features


def get_inputs_from_reverted_samples(n_inputs, means_per_dim, stds_per_dim,
                                     weights_per_cluster,
                                     feature_model,
                                     to_4d=True,
                                     gaussian_to_uniform_phases=False):
    feature_model.eval()
    sizes = sizes_from_weights(n_inputs, var_to_np(weights_per_cluster))
    gauss_samples = sample_mixture_gaussian(sizes, means_per_dim, stds_per_dim)
    if to_4d:
        gauss_samples = gauss_samples.unsqueeze(2).unsqueeze(3)
    if gaussian_to_uniform_phases:
        assert len(means_per_dim) == 1
        gauss_samples = gaussian_to_uniform_phases_in_outs(
            gauss_samples, means_per_dim.squeeze(0))
    rec_var = invert(feature_model, gauss_samples)
    rec_examples = var_to_np(rec_var).squeeze()
    return rec_examples, gauss_samples


def weights_init(module, conv_weight_init_fn):
    classname = module.__class__.__name__
    if (('Conv' in classname) or (
                'Linear' in classname)) and classname != "AvgPool2dWithConv":
        conv_weight_init_fn(module.weight)
        if module.bias is not None:
            th.nn.init.constant(module.bias, 0)
    elif 'BatchNorm' in classname:
        th.nn.init.constant(module.weight, 1)
        th.nn.init.constant(module.bias, 0)


def init_model_params(feature_model, gain):
    feature_model.apply(lambda module: weights_init(
        module,
        lambda w: th.nn.init.xavier_uniform(w, gain=gain)))

