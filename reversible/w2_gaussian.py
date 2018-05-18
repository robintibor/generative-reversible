import torch
from torch.autograd import Variable


def sqrt_newton_schulz_autograd(A, numIters, dtype):
    batchSize = A.data.shape[0]
    dim = A.data.shape[1]
    normA = A.mul(A).sum(dim=1).sum(dim=1).sqrt()
    Y = A.div(normA.view(batchSize, 1, 1).expand_as(A));
    I = Variable(torch.eye(dim,dim).view(1, dim, dim).
               repeat(batchSize,1,1).type(dtype),requires_grad=False)
    Z = Variable(torch.eye(dim,dim).view(1, dim, dim).
               repeat(batchSize,1,1).type(dtype),requires_grad=False)

    for i in range(numIters):
        T = 0.5*(3.0*I - Z.bmm(Y))
        Y = Y.bmm(T)
        Z = T.bmm(Z)
    sA = Y*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
    return sA