'''
    VQ-VAE's latent prior distribution
'''

from typing import Union, Type, Dict, Optional, Any, Tuple
from abc import abstractmethod

import torch
from torch import nn

from ..vqvae import AbstractQuantVAE

class PriorModel(nn.Module):
    '''
        Base class for prior models
    '''
    def __init__(
        self,
        in_channels : int,
        code_shape : Tuple[int, int],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.code_shape = tuple(code_shape)

    def forward(self, x, *args, **kwargs):
        ''' To be implemented '''
        raise NotImplementedError


class AbstractVQLatentPriorModel(nn.Module):
    '''
        Abstract class for VQ-VAE latent prior models
    '''
    def __init__(self, feature_extractor_model : AbstractQuantVAE, **kwargs):
        super().__init__()
        for p in feature_extractor_model.parameters():
            p.requires_grad = False

        self.feature_extractor_model = feature_extractor_model
        self.feature_extractor_model.eval()

    def to(self, device, *args, **kwargs):
        '''Override to() to ensure encoder to use correct device'''
        self.feature_extractor_model.to(device)
        super().to(device, *args, **kwargs)

    def retrieve_codes(self, x, cond):
        '''
            Retrieve discrete latent codes from VQ-VAE
            code shape is N x C_H x C_W
        '''
        self.feature_extractor_model.eval()
        with torch.no_grad():
            _, code = self.feature_extractor_model.codebook(x, cond)
        return code

    @abstractmethod
    def forward(self, x, cond = None):
        '''Retrieve codes for images'''

    @abstractmethod
    def forward_latent(self, codes, cond = None):
        '''Forward pass in latent space'''

    @abstractmethod
    def loss(
            self,
            x : torch.Tensor,
            reduction : str = 'mean',
            is_training : bool = True,
            **kwargs
        ) -> Dict[str, torch.Tensor]:
        ''' return loss, target, and prediction tensors '''

    @abstractmethod
    def _restore_abnormal(
            self,
            codes : Union[torch.Tensor, Tuple[torch.Tensor, ...]],
            logit_threshold : Union[float, Tuple[float, ...]],
            num_reconstructions : int = 1,
            is_thres_quantile : bool = False,
            n_iters : int = 1,
            **kwargs
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
            Restore from codes by removing unlikely codes
            return reconstructed images and the losses related to the orginal images
        '''

    def restore_by_codes(
        self,
        codes : Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        logit_threshold : Union[float, Tuple[float, ...]],
        num_reconstructions : int = 1,
        is_thres_quantile : bool = False,
        n_iters : int = 1,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ''' Restore by embedding codes'''
        return self._restore_abnormal(
            codes, logit_threshold,
            num_reconstructions = num_reconstructions,
            is_thres_quantile = is_thres_quantile,
            n_iters = n_iters,
            **kwargs
        )

    def restore_images(
        self,
        images : torch.Tensor,
        logit_threshold : Union[float, Tuple[float, ...]],
        num_reconstructions : int = 1,
        is_thres_quantile : bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ''' Restore images'''
        codes = self.retrieve_codes(images, None)
        return self._restore_abnormal(
            codes, logit_threshold,
            num_reconstructions = num_reconstructions,
            is_thres_quantile = is_thres_quantile,
            **kwargs
        )

    @abstractmethod
    def sample(
        self,
        num_images : int ,
        cond : Optional[Any] = None,
        image_chw : Optional[Tuple[int, int, int]] = None,
        **kwargs
    ) -> torch.Tensor:
        '''
        generates samples from the model. 'cond' is a conditional image
        '''

    def pixelwise_anomaly_score(
        self,
        images : torch.Tensor,
        logit_threshold : Union[float, Tuple[float, ...]],
        num_reconstructions : int = 1,
        is_thres_quantile : bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ''' compute pixel-wise anomaly score'''
        x_0 = self.feature_extractor_model.reconstruct(images)
        #base_kwargs = kwargs.copy()
        #base_kwargs['resample_option'] = 'normal'
        #recons, losses = self.restore_images(
        #    images, logit_threshold, num_reconstructions,
        #    is_thres_quantile = is_thres_quantile, **base_kwargs
        #)
        #x_0 = recons.mean(dim = 1)

        recons, losses = self.restore_images(
            images, logit_threshold, num_reconstructions,
            is_thres_quantile = is_thres_quantile, **kwargs
        )
        #diff = torch.abs(x_0 - recons) / torch.max(x_0, recons).clamp(min=1e-8)
        weights = losses / losses.sum()
        recon = (weights.view(weights.size(0), num_reconstructions, -1, 1, 1) * recons
            ).sum(dim = 1)
        return torch.abs(x_0 - recon), recon


class VQLatentPriorModel(AbstractVQLatentPriorModel):
    '''
        General model leanring prior in the latent space of a VQ-VAE
    '''
    def __init__(
            self,
            feature_extractor_model : AbstractQuantVAE,
            prior_model : Union[PriorModel, Type],
            code_shape : Tuple[int, int],
            **kwargs
        ):
        super().__init__(feature_extractor_model = feature_extractor_model, **kwargs)

        if isinstance(prior_model, PriorModel):
            if prior_model.in_channels != feature_extractor_model.code_size or \
                prior_model.code_shape != tuple(code_shape):
                raise ValueError('Incompatible prior model')
            self.prior_model : PriorModel = prior_model
        elif isinstance(prior_model, type):
            self.prior_model : PriorModel = prior_model(
                in_channels = feature_extractor_model.code_size,
                code_shape = code_shape,
                **kwargs
            )
        else:
            raise ValueError('"prior_model" must be nn.Module or Type')

    def to(self, device, *args, **kwargs):
        '''Override to() to ensure encoder to use correct device'''
        self.prior_model.to(device)
        super().to(device, *args, **kwargs)

    def forward(self, x, cond = None):
        '''Retrieve codes for images'''
        codes = self.retrieve_codes(x, cond)
        return self.prior_model.forward(codes, cond)

    def forward_latent(self, codes, cond = None):
        '''Forward pass in latent space'''
        return self.prior_model.forward(codes, cond)

    def sample(
        self,
        num_images : int ,
        cond : Optional[Any] = None,
        image_chw : Optional[Tuple[int, int, int]] = None,
        **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError('Sampling not implemented for general latent prior model')

    def _restore_abnormal(
        self,
        codes: torch.Tensor,
        logit_threshold: float,
        num_reconstructions: int = 1,
        is_thres_quantile: bool = False,
        n_iters: int = 1,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError('Restoration not implemented for general latent prior model')
