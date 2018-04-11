import random
import torch as th
import numpy as np

def enforce_2d(outs):
    while len(outs.size()) > 2:
        n_dims = len(outs.size())
        outs = outs.squeeze(2)
        assert len(outs.size()) == n_dims - 1
    return outs


def view_2d(outs):
    return outs.view(outs.size()[0], -1)


def ensure_on_same_device(*variables):
    any_cuda = np.any([v.is_cuda for v in variables])
    if any_cuda:
        variables = [ensure_cuda(v) for v in variables]
    return variables


def ensure_cuda(v):
    if not v.is_cuda:
        v = v.cuda()
    return v



def log_sum_exp(value, dim=None, keepdim=False):
    # https://github.com/pytorch/pytorch/issues/2591#issuecomment-338980717
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if dim is not None:
        m, _ = th.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + th.log(th.sum(th.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = th.max(value)
        sum_exp = th.sum(th.exp(value - m))
        return m + th.log(sum_exp)


def set_random_seeds(seed, cuda):
    """
    Set seeds for python random module numpy.random and torch.

    Parameters
    ----------
    seed: int
        Random seed.
    cuda: bool
        Whether to set cuda seed with torch.

    """
    random.seed(seed)
    th.manual_seed(seed)
    if cuda:
        th.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def np_to_var(X, requires_grad=False, dtype=None, pin_memory=False,
              **var_kwargs):
    """
    Convenience function to transform numpy array to `torch.autograd.Variable`.

    Converts `X` to ndarray using asarray if necessary.

    Parameters
    ----------
    X: ndarray or list or number
        Input arrays
    requires_grad: bool
        passed on to Variable constructor
    dtype: numpy dtype, optional
    var_kwargs:
        passed on to Variable constructor

    Returns
    -------
    var: `torch.autograd.Variable`
    """
    if not hasattr(X, '__len__'):
        X = [X]
    X = np.asarray(X)
    if dtype is not None:
        X = X.astype(dtype)
    X_tensor = th.from_numpy(X)
    if pin_memory:
        X_tensor = X_tensor.pin_memory()
    return th.autograd.Variable(X_tensor, requires_grad=requires_grad, **var_kwargs)


def var_to_np(var):
    """Convenience function to transform `torch.autograd.Variable` to numpy
    array.

    Should work both for CPU and GPU."""
    return var.cpu().data.numpy()


class FuncAndArgs(object):
    """Container for a function and its arguments. 
    Useful in case you want to pass a function and its arguments 
    to another function without creating a new class.
    You can call the new instance either with the apply method or 
    the ()-call operator:

    >>> FuncAndArgs(max, 2,3).apply(4)
    4
    >>> FuncAndArgs(max, 2,3)(4)
    4
    >>> FuncAndArgs(sum, [3,4])(8)
    15

    """

    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def apply(self, *other_args, **other_kwargs):
        all_args = self.args + other_args
        all_kwargs = self.kwargs.copy()
        all_kwargs.update(other_kwargs)
        return self.func(*all_args, **all_kwargs)

    def __call__(self, *other_args, **other_kwargs):
        return self.apply(*other_args, **other_kwargs)