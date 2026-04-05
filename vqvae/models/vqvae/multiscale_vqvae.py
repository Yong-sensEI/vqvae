''' Multiscale VQ-VAE module '''

from typing import Dict, Optional, Sequence, Tuple, Any

import torch
from torch import nn

from yw_basics.utils import import_object

from .base import AbstractQuantVAE
from .freq_decomposition import FrequencyDecompositionBlock

class MultiscaleQuantVAE(AbstractQuantVAE):
    '''Multiscale VQ-VAE model'''
    def __init__(
            self,
            in_channel : int,
            num_hidden : int,
            num_residual_hidden : int,
            base_vae_type : str,
            detail_vae_type : str,
            base_vae_kwargs : Dict,
            detail_vae_kwargs : Dict,
            average_kernel_size : Optional[Sequence[int]] = None
        ):
        super().__init__(in_channel = in_channel)
        self.freq_decomposition = FrequencyDecompositionBlock(
            in_channel, num_hidden, num_residual_hidden,
            average_kernel_size = average_kernel_size
        )

        self._base_vae : AbstractQuantVAE = import_object(base_vae_type)(
            in_channel = in_channel * self.freq_decomposition.num_levels,
            **base_vae_kwargs
        )
        self._detail_vae : AbstractQuantVAE = import_object(detail_vae_type)(
            in_channel = in_channel * self.freq_decomposition.num_levels,
            **detail_vae_kwargs
        )

        self._final_conv = nn.Conv2d(
            2 * self.freq_decomposition.out_dim, in_channel,
            kernel_size = 1, stride = 1
        )

    @property
    def code_size(self) -> int:
        '''Number of discrete latent codes'''
        return self._base_vae.code_size + self._detail_vae.code_size

    @property
    def base_vae(self) -> AbstractQuantVAE:
        '''Base VAE'''
        return self._base_vae

    @property
    def detail_vae(self) -> AbstractQuantVAE:
        '''Detail VAE'''
        return self._detail_vae

    def to(self, device, *args, **kwargs):
        '''Override to() to ensure all VAEs to use correct device'''
        self._base_vae.to(device)
        self._detail_vae.to(device)
        super().to(device, *args, **kwargs)

    def encode(self, x, cond=None) -> Tuple[torch.Tensor, torch.Tensor]:
        base, detail = self.freq_decomposition(x)
        z_base = self._base_vae.encode(base, cond)
        z_detail = self._detail_vae.encode(detail, cond)
        return z_base, z_detail

    def decode(self, z) -> torch.Tensor:
        z_base, z_detail = z
        base_recon = self._base_vae.decode(z_base)
        detail_recon = self._detail_vae.decode(z_detail)
        return self._final_conv(torch.cat([base_recon, detail_recon], dim = 1))

    def forward(self, x, cond = None) -> Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, Any, Any
        ]:
        assert self._base_vae.vector_quantization is not None, \
            'Base vector_quantization module is not set'
        assert self._detail_vae.vector_quantization is not None, \
            'Detail vector_quantization module is not set'
        z_base, z_detail = self.encode(x)
        loss_1, z_1, pplx_1, encodings_1, indices_1 = self._base_vae.vector_quantization(z_base)
        loss_2, z_2, pplx_2, encodings_2, indices_2 = self._detail_vae.vector_quantization(z_detail)
        x_hat = self.decode((z_1, z_2))

        return (
            loss_1 + loss_2,
            x_hat,
            pplx_1 + pplx_2,
            (encodings_1, encodings_2),
            (indices_1, indices_2)
        )

    def embed(self, indices : Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        ''' 
            Return embedding vectors from codebook indices
        '''
        base_indices, detail_indices = indices
        return (
            self._base_vae.vector_quantization.embed(base_indices),
            self._detail_vae.vector_quantization.embed(detail_indices)
        )
