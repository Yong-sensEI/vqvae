'''
    VQ-VAE model
'''

from torch import nn

from .encoder import Encoder
from .quantizer import VectorQuantizer
from .decoder import Decoder

class VQVAE(nn.Module):
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
            save_img_embedding_map = False
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
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(
            num_embeddings, embedding_dim, commitment_cost
        )
        # decode the discrete latent representation
        self.decoder = Decoder(embedding_dim, num_hidden, residual_layers, num_residual_hidden)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(num_embeddings)}
        else:
            self.img_to_embedding_map = None

    def to(self, device, *args, **kwargs):
        '''Override to() to ensure vector_quantization to use correct device'''
        self.vector_quantization.device = device
        super().to(device, *args, **kwargs)

    @property
    def code_size(self):
        '''Number of discrete latent codes'''
        return self.vector_quantization.n_embeddings

    @property
    def num_embeddings(self):
        '''number of discrete latent codes'''
        return self.code_size

    @property
    def num_hidden(self):
        '''number of channels of the last hidden layer'''
        return self.h_dim

    @property
    def num_residual_hidden(self):
        '''the hidden dimension of the residual blocks'''
        return self.res_h_dim
    @property
    def residual_layers(self):
        '''number of residual layers'''
        return self.n_res_layers

    @property
    def embedding_dim(self):
        '''dimension of the embedding vectors'''
        return self.vector_quantization.embedding_dim

    @property
    def commitment_cost(self):
        '''commitment cost'''
        return self.vector_quantization.beta

    def encode(self, x, cond = None):
        '''Encode input images to latent codes'''
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        return z_e

    def decode(self, z):
        ''' decode code'''
        return self.decode(z)

    def forward(self, x):
        '''Forward pass through VQ-VAE'''
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        embedding_loss, z_q, perplexity, _, embed_indices = self.vector_quantization(
            z_e)
        x_hat = self.decoder(z_q)

        return embedding_loss, x_hat, perplexity, embed_indices

    def codebook(self, x, cond = None):
        '''Return the codebook embeddings'''
        _, z_q, _, code = self.forward(x)
        # code shape: (batch_size*height*width, 1)
        return z_q, code
