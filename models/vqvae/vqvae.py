'''
    VQ-VAE model
'''
from typing import Union, Tuple

from .quantizer import VectorQuantizer, FiniteScalarQuantizer
from .base import QuantVAE

class VQVAE(QuantVAE):
    '''VQ-VAE model as described in
       "Neural Discrete Representation Learning"
       (https://arxiv.org/abs/1711.00937)
    '''
    def __init__(
            self,
            num_hidden,
            num_residual_hidden,
            residual_layers,
            num_embeddings,
            embedding_dim,
            commitment_cost,
            num_patches : Union[int, Tuple[int, int]] = 1,
            patch_prob : float = 0
        ):
        super().__init__(
            num_hidden,
            num_residual_hidden,
            embedding_dim,
            residual_layers,
            num_patches = num_patches,
            patch_prob = patch_prob
        )
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(
            num_embeddings, embedding_dim, commitment_cost
        )

class FSQVAE(QuantVAE):
    ''' Finite Scalar Quntizer VAE '''
    def __init__(
            self,
            num_hidden,
            num_residual_hidden,
            residual_layers,
            levels,
            embedding_dim,
            num_patches : Union[int, Tuple[int, int]] = 1,
            patch_prob : float = 0
        ):
        super().__init__(
            num_hidden,
            num_residual_hidden,
            embedding_dim,
            residual_layers,
            num_patches = num_patches,
            patch_prob = patch_prob
        )
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = FiniteScalarQuantizer(levels, embedding_dim)
