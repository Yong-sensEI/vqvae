''' CNN based decoder '''

from torch import nn
from .residual import ResidualStack


class Decoder(nn.Module):
    """
    This is the p_phi (x|z) network. Given a latent sample z p_phi 
    maps back to the original space z -> x.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim, out_channel, upsize_steps=2):
        super().__init__()
        kernel = 4
        stride = 2
        upsize_layers = []
        for i in range(upsize_steps-1):
            if i < upsize_steps-2:
                upsize_layers.append(
                    nn.ConvTranspose2d(
                        h_dim, h_dim, kernel_size=kernel,
                        stride=stride, padding=1
                    )
                )
            else:
                upsize_layers.append(
                    nn.ConvTranspose2d(h_dim, h_dim // 2, kernel_size=kernel,
                        stride=stride, padding=1
                    )
                )
            upsize_layers.append(nn.GELU())

        self.inverse_conv_stack = nn.Sequential(
            nn.ConvTranspose2d(
                in_dim, h_dim,
                kernel_size=kernel-1, stride=stride-1, padding=1
            ),
            *[nn.Conv2d(
                h_dim, h_dim,
                kernel_size=kernel-1, stride=stride-1, padding=1
            ) for _ in range(max(2 - upsize_steps, 0))],
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            *upsize_layers,
            nn.ConvTranspose2d(
                h_dim//2 if len(upsize_layers) > 0 else h_dim, out_channel,
                kernel_size=kernel, stride=stride, padding=1)
        )

    def forward(self, x):
        return self.inverse_conv_stack(x)
