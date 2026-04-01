'''
Prior model for 2-level VQ-VAE with separate transformers for base and detail codes
'''

from typing import Dict, Tuple, Union, Optional

import torch

from .base import AbstractVQLatentPriorModel
from ..vqvae.multiscale_vqvae import MultiscaleQuantVAE
from .transformer import PriorTransformer
from .transfomer_utils import (
    transformer_loss, resample_codes, flatten_codes, reconstruct_by_codes,
    create_sample_cond
)

class MultiscalePriorModel(AbstractVQLatentPriorModel):
    '''
        Prior model for 2-level VQ-VAE with separate transformers for base and detail codes
    '''
    def __init__(
            self,
            feature_extractor_model : MultiscaleQuantVAE,
            base_prior_kwargs : Dict,
            detail_prior_kwargs : Dict,
            **kwargs
        ):
        self._p_mask = kwargs.pop('mask_prob', 0.1)
        self._beta = kwargs.pop('beta', 0.01)
        self._use_tamper_mask = kwargs.pop('mask_tamper', True)

        super().__init__(
            feature_extractor_model = feature_extractor_model,
            **kwargs
        )
        self.feature_extractor_model = feature_extractor_model
        self._base_prior = PriorTransformer(
            in_channels = feature_extractor_model.base_vae.code_size,
            **base_prior_kwargs
        )
        self._detail_prior = PriorTransformer(
            in_channels = feature_extractor_model.detail_vae.code_size,
            **detail_prior_kwargs
        )

    def forward(self, x, cond = None):
        '''Retrieve codes for images'''
        base_codes, detail_codes = self.feature_extractor_model.retrieve_codes(x, cond)
        return (
            self._base_prior.forward(base_codes, cond),
            self._detail_prior.forward(detail_codes, cond)
        )

    def forward_latent(self, codes, cond = None):
        '''Forward pass in latent space'''
        base_codes, detail_codes = codes
        return (
            self._base_prior.forward(base_codes, cond),
            self._detail_prior.forward(detail_codes, cond)
        )

    def loss(
            self,
            x : torch.Tensor,
            reduction : str = 'mean',
            is_training : bool = True,
            **kwargs
        ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            base_codes, detail_codes = self.retrieve_codes(x, None)
        base_losses = transformer_loss(
            self._base_prior, base_codes,
            self._p_mask, self._beta, self._use_tamper_mask,
            reduction, is_training
        )
        detail_losses = transformer_loss(
            self._detail_prior, detail_codes,
            self._p_mask, self._beta, self._use_tamper_mask,
            reduction, is_training
        )
        n_batches = base_losses['pred'].size(0)
        return {
            'loss': base_losses['loss'] + detail_losses['loss'] if \
                reduction != 'none' else torch.cat((
                    base_losses['loss'].view(base_codes.size(0), -1),
                    detail_losses['loss'].view(base_codes.size(0), -1)
                ), dim = -1
            ),
            'pred': torch.cat((
                base_losses['pred'].view(n_batches, -1),
                detail_losses['pred'].view(n_batches, -1)
            ), dim = -1),
            'target': torch.cat((
                base_losses['target'].view(n_batches, -1),
                detail_losses['target'].view(n_batches, -1)
            ), dim = -1)
        }

    def _restore_abnormal(
        self,
        codes: Tuple[torch.Tensor, ...],
        logit_threshold: Union[float, Tuple[float, ...]],
        num_reconstructions: int = 1,
        is_thres_quantile: bool = False,
        n_iters: int = 1,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        resample_option = kwargs.pop('resample_option', 'abnormal')
        scale_option = kwargs.pop('scale_option', 'all')
        if scale_option not in ('base', 'detail', 'all'):
            raise ValueError(f'Invalid scale option: {scale_option}')

        base_codes, detail_codes = codes
        if isinstance(logit_threshold, float):
            logit_threshold = (logit_threshold, logit_threshold)

        if scale_option in ('base', 'all'):
            resampled_base_codes, flat_base_codes = resample_codes(
                self._base_prior, base_codes, logit_threshold[0],
                num_reconstructions, is_thres_quantile, n_iters,
                self._use_tamper_mask, resample_option
            )
        else:
            resampled_base_codes = flat_base_codes = flatten_codes(
                base_codes, num_reconstructions
            )
        if scale_option in ('detail', 'all'):
            resampled_detail_codes, flat_detail_codes = resample_codes(
                self._detail_prior, detail_codes, logit_threshold[1],
                num_reconstructions, is_thres_quantile, n_iters,
                self._use_tamper_mask, resample_option
            )
        else:
            resampled_detail_codes = flat_detail_codes = flatten_codes(
                detail_codes, num_reconstructions
            )

        return reconstruct_by_codes(
            self.feature_extractor_model,
            (self._base_prior, self._detail_prior),
            (resampled_base_codes, resampled_detail_codes),
            (flat_base_codes, flat_detail_codes),
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
