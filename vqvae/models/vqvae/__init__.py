from .base import AbstractQuantVAE
from .vqvae import VQVAE, FSQVAE, QuantVAE
from .multiscale_vqvae import MultiscaleQuantVAE

__all__ = ['VQVAE', 'FSQVAE', 'AbstractQuantVAE', 'QuantVAE', 'MultiscaleQuantVAE']
