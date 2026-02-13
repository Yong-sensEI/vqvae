'''
    A transformer learns sequence prior distribution
'''
from typing import Optional, Tuple

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

class PriorTransformer(nn.Module):
    ''' a transformer learns prior distribution of sequences '''
    def __init__(
            self,
            in_channels : int,
            code_shape : Tuple[int, int],
            num_heads : int,
            num_encoder_layers : int,
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
        self._pos_encoder = LearnedPositionalEncoding(
            np.prod(code_shape).tolist(),
            in_channels
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
