'''
    Base quantized VAE model
'''
from typing import Optional, Tuple, Any
from abc import abstractmethod

import torch
from torch import nn

from .encoder import Encoder
from .decoder import Decoder

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

    def forward(self, z) -> Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ]:
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

class AbstractQuantVAE(nn.Module):
    '''Abstract quantized VAE model'''
    def __init__(
            self,
            in_channel : int,
            *args, **kwargs
        ):
        super().__init__()
        self._in_channel = in_channel

    @property
    def in_dim(self) -> int:
        '''Input dimension of the model'''
        return self._in_channel

    @property
    @abstractmethod
    def code_size(self) -> int:
        '''Number of discrete latent codes'''

    def codebook(self, x, cond = None) -> Tuple[torch.Tensor, torch.Tensor]:
        '''Return the codebook embeddings'''
        _, z_q, _, _, code = self.forward(x)
        # code shape: (batch_size*height*width, 1)
        return z_q, code

    def reconstruct(self, x, cond = None) -> torch.Tensor:
        '''Reconstruct input images'''
        _, x_hat, _, _, _ = self.forward(x)
        return x_hat

    @abstractmethod
    def encode(self, x, cond = None) -> Any:
        '''Encode input images to latent codes'''

    @abstractmethod
    def decode(self, z) -> torch.Tensor:
        ''' decode code'''

    @abstractmethod
    def forward(self, x, cond = None) -> Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, Any, Any
        ]:
        '''Forward pass through VQ-VAE'''

    @abstractmethod
    def embed(self, indices : Any) -> torch.Tensor:
        ''' 
            Return an embedding vector from codebook indices
        '''

class QuantVAE(AbstractQuantVAE):
    '''
    Quantized VAE model as described in
    '''
    def __init__(
            self,
            in_channel,
            num_hidden,
            num_residual_hidden,
            embedding_dim,
            residual_layers,
            downsize_steps : int = 2,
            **kwargs
        ):
        super().__init__(in_channel = in_channel)
        self.h_dim = num_hidden
        self.res_h_dim = num_residual_hidden
        self.n_res_layers = residual_layers

        # encode image into continuous latent space
        self._encoder = Encoder(
            in_channel, num_hidden, residual_layers, num_residual_hidden, downsize_steps
        )
        self.pre_quantization_conv = nn.Conv2d(
            num_hidden, embedding_dim, kernel_size=1, stride=1
        )

        self._vector_quantization : Optional[Quantizer] = None
        # decode the discrete latent representation
        self._decoder = Decoder(
            embedding_dim, num_hidden, residual_layers,
            num_residual_hidden, in_channel, downsize_steps
        )

    def to(self, device, *args, **kwargs):
        '''Override to() to ensure vector_quantization to use correct device'''
        assert self._vector_quantization is not None, 'vector_quantization module is not set'
        self._vector_quantization.to(device)
        self._vector_quantization.device = device
        super().to(device, *args, **kwargs)

    @property
    def vector_quantization(self) -> Quantizer:
        '''Get the vector quantization module'''
        assert self._vector_quantization is not None, 'vector_quantization module is not set'
        return self._vector_quantization

    @property
    def code_size(self) -> int:
        '''Number of discrete latent codes'''
        assert self._vector_quantization is not None, 'vector_quantization module is not set'
        return self._vector_quantization.n_embeddings

    @property
    def num_embeddings(self) -> int:
        '''number of discrete latent codes'''
        return self.code_size

    def encode(self, x, cond = None):
        '''Encode input images to latent codes'''
        return self.pre_quantization_conv(self._encoder(x))

    def decode(self, z):
        ''' decode code'''
        return self._decoder(z)

    def forward(self, x, cond = None):
        '''Forward pass through VQ-VAE'''
        assert self._vector_quantization is not None, 'vector_quantization module is not set'
        z_e = self.encode(x)
        loss, z_q, perplexity, encodings, indices = self.vector_quantization(z_e)
        x_hat = self._decoder(z_q)

        return loss, x_hat, perplexity, encodings, indices

    def embed(self, indices : torch.Tensor) -> torch.Tensor:
        ''' 
            Return an embedding vector from codebook indices
            indices shape is (B, H, W)
            return (B, C, H, W) C is number of embeddings
        '''
        return self.vector_quantization.embed(indices)
