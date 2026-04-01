'''
    VQ-VAE model
'''

from .quantizer import VectorQuantizer, FiniteScalarQuantizer
from .base import QuantVAE

class VQVAE(QuantVAE):
    '''VQ-VAE model as described in
       "Neural Discrete Representation Learning"
       (https://arxiv.org/abs/1711.00937)
    '''
    def __init__(
            self,
            in_channel,
            num_hidden,
            num_residual_hidden,
            residual_layers,
            num_embeddings,
            embedding_dim,
            commitment_cost,
            downsize_steps : int = 2
        ):
        super().__init__(
            in_channel,
            num_hidden,
            num_residual_hidden,
            embedding_dim,
            residual_layers,
            downsize_steps = downsize_steps
        )
        # pass continuous latent vector through discretization bottleneck
        self._vector_quantization = VectorQuantizer(
            num_embeddings, embedding_dim, commitment_cost
        )

class FSQVAE(QuantVAE):
    ''' Finite Scalar Quntizer VAE '''
    def __init__(
            self,
            in_channel,
            num_hidden,
            num_residual_hidden,
            residual_layers,
            levels,
            embedding_dim,
            downsize_steps : int = 2,
            **kwargs
        ):
        super().__init__(
            in_channel,
            num_hidden,
            num_residual_hidden,
            embedding_dim,
            residual_layers,
            downsize_steps = downsize_steps
        )
        # pass continuous latent vector through discretization bottleneck
        self._vector_quantization = FiniteScalarQuantizer(levels, embedding_dim, **kwargs)
