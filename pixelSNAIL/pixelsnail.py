'''PixelSNAIL model implementation'''

from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F

from .nn_blocks import PixelBlock, MaskedConv, WNConv2d

class PixelSNAIL(nn.Module):
    '''
        PixelSNAIL model as described in
        "PixelSNAIL: An Improved Autoregressive Generative Model"
        (https://arxiv.org/abs/1712.09763)
    '''
    def __init__(
            self,
            in_channels,
            code_shape = (64, 64),
            num_channels = 64,
            num_blocks = 4,
            num_residual_blocks = 2,
            dropout_prob = 0.1,
            cond_channels = None,
            downsample = 1,
            non_linearity = F.elu
        ):
        super().__init__()

        self.non_linearity = non_linearity
        height, width = code_shape

        self.in_chan = in_channels
        self.cond_channels = cond_channels
        self.ini_conv = MaskedConv(
            in_channels,
            num_channels,
            kernel_size = 7,
            stride = downsample,
            mask_type = 'A'
        )

        height //= downsample
        width //= downsample

        # Creates a grid with coordinates within image
        coord_x = (torch.arange(height).float() - height / 2) / height
        coord_x = coord_x.view(1, 1, height, 1).expand(1, 1, height, width)
        coord_y = (torch.arange(width).float() - width / 2) / width
        coord_y = coord_y.view(1, 1, 1, width).expand(1, 1, height, width)
        self.register_buffer('background', torch.cat([coord_x, coord_y], 1))

        self.blocks = nn.ModuleList()

        for _ in range(num_blocks):
            self.blocks.append(
                PixelBlock(
                    num_channels, num_channels,
                    n_res_block = num_residual_blocks,
                    shape = (height, width),
                    dropout_p = dropout_prob,
                    cond_channels = cond_channels,
                    non_linearity = non_linearity
                )
            )

        self.upsample = nn.ConvTranspose2d(
            num_channels, num_channels,
            kernel_size = downsample,
            stride = downsample
        )

        self.out = WNConv2d(num_channels, in_channels, 1)

    def forward(self, x, cond=None):
        '''forward pass of PixelSNAIL'''
        x = F.one_hot( # pylint: disable=E1102
                x, self.in_chan
            ).permute(0, 3, 1, 2).type_as(self.background)

        if self.cond_channels is not None:
            cond = cond.float()

        out = self.ini_conv(x)

        batch, _, height, width = out.shape
        background = self.background.expand(batch, -1, -1, -1)

        for block in self.blocks:
            out = block(
                out, background = background, cond = cond
            )

        out = self.upsample(self.non_linearity(out))
        out = self.out(self.non_linearity(out))

        return out

    def loss(self, x, cond = None, reduction = 'mean'):
        '''computes the negative log likelihood loss'''
        logits = self.forward(x, cond)
        nll = F.cross_entropy(logits, x,reduction=reduction)
        return OrderedDict(loss=nll)

    def sample(self, n, img_size = (64,64), cond = None):
        '''generates samples from the model'''
        device = next(self.parameters()).device
        samples = torch.zeros(n, *img_size).long().to(device)
        with torch.no_grad():
            for r in range(img_size[0]):
                for c in range(img_size[1]):
                    if self.cond_channels is not None:
                        logits = self(samples,cond)[:, :, r, c]
                    else:
                        logits = self(samples)[:, :, r, c]
                    probs = F.softmax(logits, dim=1)
                    samples[:, r, c] = torch.multinomial(probs, 1).squeeze(-1)
        return samples.cpu().numpy()
