'''
    Base quantized VAE model
'''
from typing import Optional, Tuple, Union
from abc import abstractmethod

import torch
from torch import nn

from .encoder import Encoder
from .decoder import Decoder
from .. import mask_by_patches

class Quantizer(nn.Module):
    '''Quantizer module'''
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._device = None

    @property
    def device(self):
        '''Get device of this module'''
        return self._device

    @device.setter
    def device(self, value):
        '''Set device of this module'''
        self._device = value

    @abstractmethod
    def embed(self, indices : torch.Tensor) -> torch.Tensor:
        ''' 
            Return an embedding vector from codebook indices
            indices shape is (B, H, W)
            return (B, C, H, W) C is number of embeddings
        '''

    def forward(self, z):
        '''Forward pass through quantizer'''
        raise NotImplementedError('forward method not implemented in Quantizer')

    @property
    @abstractmethod
    def n_embeddings(self) -> int:
        '''Number of discrete latent codes'''

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        '''Dimension of the embedding vectors'''

class QuantVAE(nn.Module):
    '''
    Quantized VAE model as described in
    '''
    def __init__(
            self,
            num_hidden,
            num_residual_hidden,
            embedding_dim,
            residual_layers,
            num_patches : Union[int, Tuple[int, int]] = 1,
            patch_prob : float = 0,
            **kwargs
        ):
        super().__init__()
        self.h_dim = num_hidden
        self.res_h_dim = num_residual_hidden
        self.n_res_layers = residual_layers

        # encode image into continuous latent space
        self.encoder = Encoder(3, num_hidden, residual_layers, num_residual_hidden)
        self.pre_quantization_conv = nn.Conv2d(
            num_hidden, embedding_dim, kernel_size=1, stride=1
        )

        self.vector_quantization : Optional[Quantizer] = None
        # decode the discrete latent representation
        self.decoder = Decoder(embedding_dim, num_hidden, residual_layers, num_residual_hidden)
        self._n_patches : Tuple[int, int] = num_patches if isinstance(num_patches, (tuple, list)) \
            else (num_patches, num_patches)
        self._patch_prob = patch_prob

    def to(self, device, *args, **kwargs):
        '''Override to() to ensure vector_quantization to use correct device'''
        assert self.vector_quantization is not None, 'vector_quantization module is not set'
        self.vector_quantization.to(device)
        self.vector_quantization.device = device
        super().to(device, *args, **kwargs)

    @property
    def code_size(self) -> int:
        '''Number of discrete latent codes'''
        assert self.vector_quantization is not None, 'vector_quantization module is not set'
        return self.vector_quantization.n_embeddings

    @property
    def num_embeddings(self) -> int:
        '''number of discrete latent codes'''
        return self.code_size

    @property
    def num_hidden(self) -> int:
        '''number of channels of the last hidden layer'''
        return self.h_dim

    @property
    def num_residual_hidden(self) -> int:
        '''the hidden dimension of the residual blocks'''
        return self.res_h_dim

    @property
    def residual_layers(self) -> int:
        '''number of residual layers'''
        return self.n_res_layers

    @property
    def embedding_dim(self) -> int:
        '''dimension of the embedding vectors'''
        assert self.vector_quantization is not None, 'vector_quantization module is not set'
        return self.vector_quantization.embedding_dim

    def encode(self, x, cond = None):
        '''Encode input images to latent codes'''
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        return z_e

    def decode(self, z):
        ''' decode code'''
        return self.decode(z)

    def forward(self, x, cond = None):
        '''Forward pass through VQ-VAE'''
        assert self.vector_quantization is not None, 'vector_quantization module is not set'
        z_e = self.encoder(self._mask_by_patches(x))
        z_e = self.pre_quantization_conv(z_e)
        loss, z_q, perplexity, encodings, indices = self.vector_quantization(z_e)
        x_hat = self.decoder(z_q)

        return loss, x_hat, perplexity, encodings, indices

    def codebook(self, x, cond = None):
        '''Return the codebook embeddings'''
        _, z_q, _, _, code = self.forward(x)
        # code shape: (batch_size*height*width, 1)
        return z_q, code

    def reconstruct(self, x, cond = None):
        '''Reconstruct input images'''
        _, x_hat, _, _, _ = self.forward(x)
        return x_hat

    def _mask_by_patches(self, x):
        '''Mask input images by patches'''
        return mask_by_patches(x, self._n_patches, self._patch_prob)
