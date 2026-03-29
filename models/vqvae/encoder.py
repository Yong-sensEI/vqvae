''' CNN based encoder '''

import torch
from torch import nn
import numpy as np
from .residual import ResidualStack


class Encoder(nn.Module):
    """
    This is the q_theta (z|x) network. Given a data sample x q_theta 
    maps to the latent space x -> z.

    For a VQ VAE, q_theta outputs parameters of a categorical distribution.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim, downsize_steps = 2):
        super().__init__()
        kernel = 4
        stride = 2
        downsize_layers = []
        for i in range(downsize_steps):
            if i == 0:
                downsize_layers.append(
                    nn.Conv2d(
                        in_dim, h_dim // 2, kernel_size=kernel,
                        stride=stride, padding=1
                    )
                )
            elif i == downsize_steps - 1:
                downsize_layers.append(
                    nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel,
                        stride=stride, padding=1
                    )
                )
            else:
                downsize_layers.append(
                    nn.Conv2d(h_dim // 2, h_dim // 2, kernel_size=kernel,
                        stride=stride, padding=1
                    )
                )
            downsize_layers.append(nn.GELU())

        self.conv_stack = nn.Sequential(
            *downsize_layers,
            nn.Conv2d(h_dim, h_dim, kernel_size=kernel-1,
                      stride=stride-1, padding=1),
            ResidualStack(
                h_dim, h_dim, res_h_dim, n_res_layers)
        )

    def forward(self, x):
        return self.conv_stack(x)
