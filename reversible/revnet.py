### Reversible model parts

import copy
import math

import torch as th
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from reversible.ampphase import amp_phase_to_x_y
from reversible.ampphase import gaussian_to_uniform_phases_in_outs
from reversible.gaussian import sizes_from_weights, sample_mixture_gaussian, \
    get_gauss_samples
from reversible.uniform import get_uniform_samples
from reversible.util import var_to_np


def get_all_outs(feature_model, inputs):
    """Assumes sequential model"""
    all_outs = [inputs]
    cur_out = inputs
    for module in feature_model.children():
        cur_out = module(cur_out)
        all_outs.append(cur_out)
    return all_outs


# from https://github.com/silvandeleemput/memcnn/blob/242478b9a55cd3617e8b795a8f6f5653a1ee485d/memcnn/models/revop.py
class ReversibleBlock(nn.Module):
    def __init__(self, Fm, Gm=None, implementation=0, keep_input=False):
        super(ReversibleBlock, self).__init__()
        # mirror the passed module, without parameter sharing...
        if Gm is None:
            Gm = copy.deepcopy(Fm)
        self.Gm = Gm
        self.Fm = Fm
        self.implementation = implementation
        self.keep_input = keep_input

    def forward(self, x):
        # These functions should not store their activations during training (train mode),
        # but the weights need updates on the backward pass
        args = [x, self.Fm, self.Gm] + [w for w in self.Fm.parameters()] + [w for w in self.Gm.parameters()]
        if self.implementation == 0:
            out = ReversibleBlockFunction.apply(*args)
        elif self.implementation == 1:
            out = ReversibleBlockFunction2.apply(*args)
        else:
            raise NotImplementedError("Selected implementation ({}) not implemented...".format(self.implementation))

        # Clears the input data as it can be reversed on the backward pass
        if not self.keep_input:
            x.data.set_()

        return out

class ReversibleBlockFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, Fm, Gm, *weights):
        """Forward pass for the reversible block computes:
        {x1, x2} = x
        y1 = x1 + Fm(x2)
        y2 = x2 + Gm(y1)
        output = {y1, y2}
        Parameters
        ----------
        ctx : torch.autograd.function.RevNetFunctionBackward
            The backward pass context object
        x : TorchTensor
            Input tensor. Must have channels (2nd dimension) that can be partitioned in two equal partitions
        Fm : nn.Module
            Module to use for computation, must retain dimensions such that Fm(X)=Y, X.shape == Y.shape
        Gm : nn.Module
            Module to use for computation, must retain dimensions such that Gm(X)=Y, X.shape == Y.shape
        *weights : TorchTensor
            weights for Fm and Gm in that order {Fm_w1, ... Fm_wn, Gm_w1, ... Gm_wn}
        Note
        ----
        All tensor/autograd variable input arguments and the output are
        TorchTensors for the scope of this fuction
        """
        # check if possible to partition into two equally sized partitions
        assert(x.shape[1] % 2 == 0) # assert if possible
        partition = x.shape[1] / 2

        # store partition size, Fm and Gm functions in context
        ctx.Fm = Fm
        ctx.Gm = Gm

        # partition in two equally sized set of channels
        #x1, x2 = torch.chunk(x, 2, dim=1)
        # HACK robintibor@gmail.com make same as mine
        x2, x1 = torch.chunk(x, 2, dim=1)
        x1, x2 = x1.contiguous(), x2.contiguous()

        # compute outputs
        x2var = Variable(x2, requires_grad=False, volatile=True)
        fmr = Fm.forward(x2var).data

        y1 = x1 + fmr
        x1.set_()
        del x1
        y1var = Variable(y1, requires_grad=False, volatile=True)
        gmr = Gm.forward(y1var).data
        y2 = x2 + gmr
        x2.set_()
        del x2
        output = torch.cat([y1, y2], dim=1)
        y1.set_()
        y2.set_()
        del y1, y2

        # save the input and output variables
        ctx.save_for_backward(x, output)

        return output


    @staticmethod
    def backward(ctx, grad_output):

        Fm, Gm = ctx.Fm, ctx.Gm
        # are all variable objects now
        x, output = ctx.saved_variables #[0]
        y1, y2 = Variable.chunk(output, 2, dim=1)
        y1, y2 = y1.contiguous(), y2.contiguous()

        # partition output gradient also on channels
        assert(grad_output.data.shape[1] % 2 == 0)
        y1_grad, y2_grad = Variable.chunk(grad_output, 2, dim=1)
        y1_grad, y2_grad = y1_grad.contiguous(), y2_grad.contiguous()

        # Recreate computation graphs for functions Gm and Fm with gradient collecting leaf nodes:
        # z1_stop, x2_stop, GW, FW
        # Also recompute inputs (x1, x2) from outputs (y1, y2)
        z1_stop = Variable(y1.data, requires_grad=True)

        GWeights = [p for p in Gm.parameters()]
        G_z1 = Gm.forward(z1_stop)
        x2 = y2 - G_z1
        x2_stop = Variable(x2.data, requires_grad=True)

        FWeights = [p for p in Fm.parameters()]
        F_x2 = Fm.forward(x2_stop)
        x1 = y1 - F_x2
        x1_stop = Variable(x1.data, requires_grad=True)

        # Compute outputs building a sub-graph
        z1 = x1_stop + F_x2
        y2_ = x2_stop + G_z1
        y1_ = z1

        # Perform full backward pass on graph...
        y = torch.cat([y1_, y2_], dim=1)
        dd = torch.autograd.grad(y, (x1_stop, x2_stop) + tuple(Gm.parameters()) + tuple(Fm.parameters()), grad_output, retain_graph=False)
        GWgrads = dd[2:2+len(GWeights)]
        FWgrads = dd[2+len(GWeights):]
        x2_grad = dd[1]
        x1_grad = dd[0]

        #HACK change order
        #grad_input = torch.cat([x1_grad, x2_grad], dim=1)
        grad_input = torch.cat([x2_grad, x1_grad], dim=1)

        y1_.detach_()
        y2_.detach_()
        del y1_, y2_

        # restore input
        #HACK change order
        #x.data.set_(torch.cat([x1.data, x2.data], dim=1).contiguous())
        x.data.set_(torch.cat([x2.data, x1.data], dim=1).contiguous())
        return (grad_input, None, None) + FWgrads + GWgrads


class ReversibleBlockFunction2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, Fm, Gm, *weights):
        """Forward pass for the reversible block computes:
        {x1, x2} = x
        y1 = x1 + Fm(x2)
        y2 = x2 + Gm(y1)
        output = {y1, y2}
        Parameters
        ----------
        ctx : torch.autograd.function.RevNetFunctionBackward
            The backward pass context object
        x : TorchTensor
            Input tensor. Must have channels (2nd dimension) that can be partitioned in two equal partitions
        Fm : nn.Module
            Module to use for computation, must retain dimensions such that Fm(X)=Y, X.shape == Y.shape
        Gm : nn.Module
            Module to use for computation, must retain dimensions such that Gm(X)=Y, X.shape == Y.shape
        *weights : TorchTensor
            weights for Fm and Gm in that order {Fm_w1, ... Fm_wn, Gm_w1, ... Gm_wn}
        Note
        ----
        All tensor/autograd variable input arguments and the output are
        TorchTensors for the scope of this fuction
        """
        # check if possible to partition into two equally sized partitions
        assert(x.shape[1] % 2 == 0) # assert if possible
        partition = x.shape[1] / 2

        # store partition size, Fm and Gm functions in context
        ctx.Fm = Fm
        ctx.Gm = Gm

        # partition in two equally sized set of channels
        #x1, x2 = torch.chunk(x, 2, dim=1)
        # Hack change order
        x2, x1 = torch.chunk(x, 2, dim=1)
        x1, x2 = x1.contiguous(), x2.contiguous()

        # compute outputs
        x2var = Variable(x2, requires_grad=False, volatile=True)
        fmr = Fm.forward(x2var).data

        y1 = x1 + fmr
        x1.set_()
        del x1
        y1var = Variable(y1, requires_grad=False, volatile=True)
        gmr = Gm.forward(y1var).data
        y2 = x2 + gmr
        x2.set_()
        del x2
        output = torch.cat([y1, y2], dim=1)
        y1.set_()
        del y1
        y2.set_()
        del y2

        # save the input and output variables
        ctx.save_for_backward(x, output)

        return output


    @staticmethod
    def backward(ctx, grad_output):

        Fm, Gm = ctx.Fm, ctx.Gm
        # are all variable objects now
        x, output = ctx.saved_variables #[0]
        y1, y2 = Variable.chunk(output, 2, dim=1)
        y1, y2 = y1.contiguous(), y2.contiguous()

        # partition output gradient also on channels
        assert(grad_output.data.shape[1] % 2 == 0)
        y1_grad, y2_grad = Variable.chunk(grad_output, 2, dim=1)
        y1_grad, y2_grad = y1_grad.contiguous(), y2_grad.contiguous()

        # Recreate computation graphs for functions Gm and Fm with gradient collecting leaf nodes:
        # z1_stop, x2_stop, GW, FW
        # Also recompute inputs (x1, x2) from outputs (y1, y2)

        z1_stop = Variable(y1.data, requires_grad=True)

        GWeights = [p for p in Gm.parameters()]
        G_z1 = Gm.forward(z1_stop)
        x2 = y2 - G_z1
        x2_stop = Variable(x2.data, requires_grad=True)

        FWeights = [p for p in Fm.parameters()]
        F_x2 = Fm.forward(x2_stop)
        x1 = y1 - F_x2
        x1_stop = Variable(x1.data, requires_grad=True)

        # Compute outputs building a sub-graph
        z1 = x1_stop + F_x2
        y2_ = x2_stop + G_z1
        y1_ = z1

        # Calculate the final gradients for
        dd = torch.autograd.grad(y2_, (z1_stop,) + tuple(Gm.parameters()), y2_grad, retain_graph=False)
        z1_grad = dd[0] + y1_grad
        GWgrads = dd[1:]

        dd = torch.autograd.grad(y1_, (x1_stop, x2_stop) + tuple(Fm.parameters()), y2_grad, retain_graph=False)

        FWgrads = dd[2:]
        x2_grad = dd[1] + y2_grad
        x1_grad = dd[0]
        # hack change order
        #grad_input = torch.cat([x1_grad, x2_grad], dim=1)
        grad_input = torch.cat([x2_grad, x1_grad], dim=1)

        y1_.detach_()
        y2_.detach_()
        del y1_, y2_

        # restore input
        #x.data.set_(torch.cat([x1.data, x2.data], dim=1).contiguous())
        # hack change order
        x.data.set_(torch.cat([x2.data, x1.data], dim=1).contiguous())
        return (grad_input, None, None) + FWgrads + GWgrads


class ReversibleBlockOld(th.nn.Module):
    def __init__(self, F, G):
        super(ReversibleBlockOld, self).__init__()
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


def invert(feature_model, features, return_all=False):
    if (feature_model.__class__.__name__ == 'ReversibleBlock') or \
            (feature_model.__class__.__name__ == 'SubsampleSplitter') or (
                feature_model.__class__.__name__ == 'ReversibleBlockOld'
    ):
        feature_model = th.nn.Sequential(feature_model, )
    all_features = []
    all_features.append(features)
    for module in reversed(list(feature_model.children())):
        if module.__class__.__name__ == 'ReversibleBlockOld':
            n_chans = features.size()[1]
            # y1 = self.F(x1) + x2
            # y2 = self.G(y1) + x1
            y1 = features[:, :n_chans // 2]
            y2 = features[:, n_chans // 2:]

            x1 = y2 - module.G(y1)
            x2 = y1 - module.F(x1)
            features = th.cat((x1, x2), dim=1)
        if module.__class__.__name__ == 'ReversibleBlock':
            n_chans = features.size()[1]
            # y1 = self.F(x1) + x2
            # y2 = self.G(y1) + x1
            y1 = features[:, :n_chans // 2]
            y2 = features[:, n_chans // 2:]
            x1 = y2 - module.Gm(y1)
            x2 = y1 - module.Fm(x1)
            # OLD, No longer correct:
            #y1 = F(x2) + x1
            #y2 = G(y1) + x2
            #x2 = y2 - module.Gm(y1)
            #x1 = y1 - module.Fm(x2)
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
        if return_all:
            all_features.append(features)
    if return_all:
        return all_features
    else:
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


def get_inputs_from_gaussian_samples(n_inputs, mean, std,
                                     feature_model):
    gauss_samples = get_gauss_samples(n_inputs, mean, std)
    inverted = invert(feature_model, gauss_samples)
    rec_examples = var_to_np(inverted).squeeze()
    return rec_examples, gauss_samples


def get_inputs_from_uniform_samples(n_inputs, mean, std,
                                     feature_model):
    uniform_samples = get_uniform_samples(n_inputs, mean, std)
    inverted = invert(feature_model, uniform_samples)
    rec_examples = var_to_np(inverted).squeeze()
    return rec_examples, uniform_samples


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

