'''
    Vector Quantizer for VQ-VAE
'''
from typing import Optional, Sequence

import numpy as np
import torch
from torch import nn

from .base import Quantizer

class VectorQuantizer(Quantizer):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_embeddings, embedding_dim, beta):
        super().__init__()
        self._n_embeddings = n_embeddings
        self._embedding_dim = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(
            -1.0 / self.n_embeddings, 1.0 / self.n_embeddings
        )

        self._device = self.embedding.weight.device

    @property
    def n_embeddings(self) -> int:
        '''Number of discrete latent codes'''
        return self._n_embeddings

    @property
    def embedding_dim(self) -> int:
        '''Dimension of the embedding vectors'''
        return self._embedding_dim

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)

        """
        b, c, h, w = z.shape

        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.embedding_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_embeddings
        ).to(self.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        min_encodings = min_encodings.view(
            b, h, w, self.n_embeddings
        ).permute(0, 3, 1, 2).contiguous()

        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return (
            loss, z_q, perplexity, min_encodings,
            min_encoding_indices.squeeze_().view(b, h, w)
        )

    def embed(self, indices : torch.Tensor) -> torch.Tensor:
        ''' 
            Return an embedding vector from codebook indices
            indices shape is (B, H, W)
            return (B, C, H, W) C is number of embeddings
        '''
        return self.embedding(indices).permute(0, 3, 1, 2).contiguous()

class FiniteScalarQuantizer(Quantizer):
    ''' 
    Finite scalar quantizer
    https://github.com/google-research/google-research/tree/master/fsq
    '''
    def __init__(
            self,
            levels : Sequence[int],
            embedding_dim : Optional[int] = None,
            bound_function : str = 'ifsq'
        ):
        super().__init__()

        self._cb_size = np.prod(levels).tolist()
        self._n_levels = len(levels)
        if embedding_dim is not None:
            self._embedding_dim = embedding_dim
            self._embd_in = nn.Linear(self._embedding_dim, self._n_levels)
            self._embd_out = nn.Linear(self._n_levels, self._embedding_dim)
        else:
            self._embedding_dim = self._n_levels
            self._embd_in = nn.Identity()
            self._embd_out = nn.Identity()

        self.register_buffer('_levels', torch.tensor(levels), persistent = False)
        self.register_buffer(
            '_half_width', self._levels // 2, persistent = False  # type: ignore
        )
        self.register_buffer(
            '_half_l',
            (self._levels - 1) * (1 - 1e-6) / 2, # type: ignore
            persistent = False
        )
        self.register_buffer(
            '_offset',
            torch.where(self._levels % 2 == 1, 0.0, 0.5), # type: ignore
            persistent = False
        )
        self.register_buffer(
            '_shift',
            torch.tan(self._offset / self._half_l), # type: ignore
            persistent = False
        )
        self.register_buffer(
            '_basis',
            torch.from_numpy(
                np.concatenate(([1,], np.cumprod(self._levels[:-1]))) # type: ignore
            ),
            persistent = False
        )
        self.register_buffer(
            '_implicit_codebook',
            self.indices_to_codes(torch.arange(0, self.n_embeddings)),
            persistent = False
        )

        if bound_function.lower() == 'tanh':
            self._bound_func = torch.tanh
        elif bound_function.lower() == 'ifsq':
            self._bound_func = lambda x: 2.0 * torch.sigmoid(1.6 * x) - 1
        else:
            raise ValueError(f"Invalid bound function: {bound_function}")

        self._device = self._levels.device

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def n_levels(self) -> int:
        ''' FSQ dim '''
        return self._n_levels

    @property
    def n_embeddings(self) -> int:
        return self._cb_size

    @staticmethod
    def round_ste(z: torch.Tensor) -> torch.Tensor:
        """Round with straight through gradients."""
        zhat = torch.round(z)
        return z + (zhat - z).detach()

    def _scale_and_shift(self, z: torch.Tensor):
        # Scale and shift to range [0, ..., L-1]
        return (z * self._half_width) + self._half_width # type: ignore

    def _scale_and_shift_inverse(self, z: torch.Tensor):
        ''' reverse scale&shift '''
        return (z - self._half_width) / self._half_width # type: ignore

    def codes_to_indices(self, codes: torch.Tensor) -> torch.Tensor:
        """Converts a `code` to an index in the codebook."""
        assert codes.shape[-1] == self.n_levels
        zhat = self._scale_and_shift(codes)
        return (zhat * self._basis).sum(dim = -1).to(torch.long) # type: ignore

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """Inverse of `codes_to_indices`."""
        indices = indices.unsqueeze(-1)
        codes_non_centered = torch.remainder(
            torch.floor_divide(indices, self._basis), # type: ignore
            self._levels # type: ignore
        ) # type: ignore
        return self._scale_and_shift_inverse(codes_non_centered)

    def bound(self, z: torch.Tensor) -> torch.Tensor:
        """Bound `z`, an array of shape (..., d)."""
        return self._bound_func(z + self._shift) * self._half_l - self._offset

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Quanitzes z, returns quantized zhat, same shape as z."""
        quantized = self.round_ste(self.bound(z))

        # Renormalize to [-1, 1].
        return quantized / self._half_width # type: ignore

    def forward(self, z):
        b, _, h, w = z.shape

        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        codes = self.quantize(self._embd_in(z.view(-1, self.embedding_dim)))
        encodings = self._embd_out(codes).view(z.shape).permute(0, 3, 1, 2).contiguous()
        encoding_indices = self.codes_to_indices(codes).view(b, h, w)

        return (
            torch.zeros(1).to(self.device), # loss
            encodings, # quantized latent vec
            torch.zeros(1).to(self.device),
            codes.view(b, h, w, self.n_levels).permute(0, 3, 1, 2).contiguous(),
            encoding_indices
        )

    def embed(self, indices : torch.Tensor) -> torch.Tensor:
        ''' 
            Return an embedding vector from codebook indices
            indices shape is (B, H, W)
            return (B, C, H, W) C is number of embeddings
        '''
        b, h, w = indices.shape
        return self._embd_out(
            self.indices_to_codes(indices.view(b, -1))
        ).view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
