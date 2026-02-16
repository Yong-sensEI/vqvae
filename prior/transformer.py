'''
    A transformer learns sequence prior distribution
'''
from typing import Optional, Tuple
import math

import numpy as np
import torch
from torch import nn

class LearnedPositionalEncoding(nn.Module):
    ''' For positional embedding '''
    def __init__(self, max_seq_len : int, dim : int):
        super().__init__()
        self._position_embeddings = nn.Embedding(max_seq_len, dim)
        self._seq_len = max_seq_len

    def forward(self, x):
        ''' forward function '''
        positions = torch.arange(
            self._seq_len,
            device = x.device
        ).expand(x.size(0), -1)
        return x + self._position_embeddings(positions)

class SinusoidalPositionalEmbedding(nn.Module):
    '''
    Sinusoidal positional embedding as described in "Attention is All You Need"
    '''
    POS_CONST = 1e4

    def __init__(self, max_seq_len : int, dim : int):
        super().__init__()
        # Compute positional encodings in log space for numerical stability
        pe = torch.zeros(max_seq_len, dim)
        position = torch.arange(0, max_seq_len, dtype = float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(self.POS_CONST) / dim)
        )

        # Apply sin to even indices (0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd indices (1, 3, 5, ...)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (non-trainable, but part of state_dict)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        ''' x shape: [batch_size, seq_len, dim] '''
        return x + self.pe[:, :x.size(1), :]

class PriorTransformer(nn.Module):
    ''' a transformer learns prior distribution of sequences '''
    def __init__(
            self,
            in_channels : int,
            code_shape : Tuple[int, int],
            num_heads : int,
            num_encoder_layers : int,
            positional_encoding : str = 'sinusoidal',
            **kwargs
        ):
        super().__init__()
        _encoder = nn.TransformerEncoderLayer(
            d_model = in_channels,
            nhead = num_heads,
            batch_first = True,
            **kwargs
        )
        self._encoders = nn.TransformerEncoder(
            _encoder,
            num_layers = num_encoder_layers,
            enable_nested_tensor = False
        )
        if positional_encoding == 'learned':
            self._pos_encoder = LearnedPositionalEncoding(
                np.prod(code_shape).tolist(),
                in_channels
            )
        elif positional_encoding == 'none':
            self._pos_encoder = nn.Identity()
        elif positional_encoding == 'sinusoidal':
            self._pos_encoder = SinusoidalPositionalEmbedding(
                np.prod(code_shape).tolist(),
                in_channels
            )
        else:
            raise ValueError(
                f'invalid positional encoding type: {positional_encoding}'
            )

    def forward(
            self,
            x : torch.Tensor,
            mask : Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        ''' forward function '''
        return self._encoders.forward(
            self._pos_encoder(x),
            mask = None,
            src_key_padding_mask = mask,
            is_causal = False
        )
