'''
frequqncy decomposition module for VQ-VAE
Algorithm from
"Neural Image Compression via Attentional Multi-scale Back Projection and Frequency Decomposition"
(https://arxiv.org/abs/2206.04634) G. Gao et al., ICCV 2021
'''

from typing import Optional, Sequence

import torch
from torch import nn

from .residual import ResidualLayer

class FrequencyDecompositionBlock(nn.Module):
    '''Decompose input into a base (low-f) layer and a detail (high-f) layer '''
    def __init__(
            self,
            in_dim : int,
            h_dim : int,
            res_h_dim : int,
            average_kernel_size : Optional[Sequence[int]] = None
        ):
        super().__init__()
        self._in_dim = in_dim
        avg_ker = [
            2 * (k//2) + 1 for k in average_kernel_size or (3, 5, 7)
        ]
        self._pools = [
            nn.AvgPool2d(kernel_size = k, padding = (k-1)//2, stride = 1)
            for k in avg_ker
        ]
        res_in_dim = len(avg_ker) * in_dim
        self._base_res = ResidualLayer(res_in_dim, h_dim, res_h_dim, res_in_dim)
        self._detail_res = ResidualLayer(res_in_dim, h_dim, res_h_dim, res_in_dim)

    @property
    def num_levels(self) -> int:
        '''Number of frequency levels'''
        return len(self._pools)

    @property
    def out_dim(self) -> int:
        '''Output dimension of the block'''
        return self._in_dim * self.num_levels

    def forward(self, x):
        avgs = [pool(x) for pool in self._pools]
        base = torch.cat(avgs, dim = 1)
        detail = torch.cat([x - avg for avg in avgs], dim = 1)
        return self._base_res(base), self._detail_res(detail)
