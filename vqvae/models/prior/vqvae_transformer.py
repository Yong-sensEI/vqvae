'''
    Transformer to learn VQ-VAE's latent prior distribution
'''

from typing import Optional, Tuple

import torch
from torch.nn import functional as F

from .base import VQLatentPriorModel
from .transformer import PriorTransformer
from ..vqvae import QuantVAE
from .transfomer_utils import (
    to_onehot, transformer_loss, resample_codes, reconstruct_by_codes,
    create_sample_cond
)

class VQLatentTransformer(VQLatentPriorModel):
    '''
        Transformer model learning prior in the latent space of a VQ-VAE
    '''
    def __init__(self, feature_extractor_model : QuantVAE, **kwargs):
        self._p_mask = kwargs.pop('mask_prob', 0.1)
        self._beta = kwargs.pop('beta', 0.01)
        self._use_tamper_mask = kwargs.pop('mask_tamper', True)
        super().__init__(
            feature_extractor_model,
            prior_model = PriorTransformer,
            **kwargs
        )

    def to_onehot(self, codes : torch.Tensor) -> torch.Tensor:
        ''' convert codes to one-hot tensors '''
        return to_onehot(
            codes,
            code_size = self.feature_extractor_model.code_size
        )

    def loss(
            self,
            x : torch.Tensor,
            reduction : str = 'mean',
            is_training : bool = True,
            **kwargs
        ):
        ''' loss function for training '''
        if is_training:
            assert reduction in ('mean', 'sum'), 'invalid reduction method'

        with torch.no_grad():
            codes = self.retrieve_codes(x, None)

        return transformer_loss(
            self.prior_model, codes, # type: ignore
            self._p_mask, self._beta, self._use_tamper_mask,
            reduction, is_training,
            **kwargs
        )

    def _restore_abnormal(
            self,
            codes : torch.Tensor,
            logit_threshold : float,
            num_reconstructions : int = 1,
            is_thres_quantile : bool = False,
            n_iters : int = 1,
            **kwargs
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        resample_option = kwargs.pop('resample_option', 'abnormal')
        flat_codes, orig_flat_codes = resample_codes(
            self.prior_model, codes, # type: ignore
            logit_threshold,
            num_reconstructions,
            is_thres_quantile,
            n_iters,
            self._use_tamper_mask,
            resample_option
        )

        return reconstruct_by_codes(
            self.feature_extractor_model,
            self.prior_model, # type: ignore
            flat_codes, orig_flat_codes,
            num_reconstructions,
             **kwargs
        )

    def sample(
            self,
            num_images : int,
            cond : Optional[torch.Tensor] = None,
            image_chw : Optional[Tuple[int, int, int]] = None,
            **kwargs
        ):
        ''' sample images from the prior model, optionally conditioned on input images '''
        dev = next(self.parameters()).device
        cond, n_iters, loss_quantile, kwargs = create_sample_cond(
            image_chw, cond, num_images, dev, kwargs
        )

        with torch.no_grad():
            codes = self.retrieve_codes(cond, None)

        return self._restore_abnormal(
            codes,
            logit_threshold = loss_quantile,
            num_reconstructions = 1,
            is_thres_quantile = True,
            n_iters = n_iters,
            **kwargs
        )[0]

    def mix_sample(self, num_images : int, conds : torch.Tensor, **kwargs):
        ''' 
        mix multiple conditional images to generate new images
        conds: conditional images of shape (num_conds, C, H, W)
        '''
        dev = next(self.parameters()).device

        with torch.no_grad():
            codes = self.retrieve_codes(conds.to(dev), None)
            flat_codes = codes.view(codes.size(0), -1)
            probs = F.softmax(
                self.prior_model.forward(self.to_onehot(flat_codes), None),
                dim = -1,
            ).mean(dim = 0)
            mix_codes = torch.multinomial(
                probs, num_images, replacement = True
            ).view(codes.size(1), codes.size(2), num_images).permute(2, 0, 1)

        return self._restore_abnormal(
            mix_codes,
            logit_threshold = kwargs.pop('loss_quantile', 0.9),
            num_reconstructions = 1,
            is_thres_quantile = True,
            n_iters = int(kwargs.pop('n_iters', 1)),
            resample_option = kwargs.pop('resample_option', 'abnormal'),
            **kwargs
        )[0]
