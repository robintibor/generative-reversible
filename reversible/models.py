import torch.nn as nn
from reversible.revnet import ReversibleBlockOld, SubsampleSplitter, ViewAs



def rev_block(n_chans, n_intermediate_chans, kernel_size=3):
    c = n_chans // 2
    n_i_c = n_intermediate_chans
    assert kernel_size % 2 == 1
    padding = kernel_size // 2
    return ReversibleBlockOld(
        nn.Sequential(
            nn.Conv2d(c, n_i_c, (kernel_size, kernel_size), padding=padding),
             nn.ReLU(),
             nn.Conv2d(n_i_c, c, (kernel_size,kernel_size), padding=padding)),
        nn.Sequential(
            nn.Conv2d(c, n_i_c, (kernel_size,kernel_size), padding=padding),
            nn.ReLU(),
            nn.Conv2d(n_i_c, c, (kernel_size,kernel_size), padding=padding)))


def create_celebA_model():
    feature_model = nn.Sequential(
        SubsampleSplitter(stride=2, checkerboard=True, chunk_chans_first=False),  # 1
        rev_block(4 * 3, 25),
        rev_block(4 * 3, 25),
        SubsampleSplitter(stride=2, checkerboard=True),  # 4
        rev_block(16 * 3, 50),
        rev_block(16 * 3, 50),
        SubsampleSplitter(stride=2, checkerboard=True),  # 7
        rev_block(64 * 3, 100),
        rev_block(64 * 3, 100),
        SubsampleSplitter(stride=2, checkerboard=True),  # 10
        rev_block(256 * 3, 200),
        rev_block(256 * 3, 200),
        SubsampleSplitter(stride=2, checkerboard=True),  # 13
        rev_block(1024 * 3, 400),
        rev_block(1024 * 3, 400),
        SubsampleSplitter(stride=2, checkerboard=True),  # 16
        rev_block(4096 * 3, 400, kernel_size=1),
        ViewAs((-1, 4096 * 3, 1, 1), (-1, 4096 * 3)), )
    return feature_model