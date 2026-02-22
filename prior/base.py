'''
    VQ-VAE's latent prior distribution
'''

from typing import Union, Type, Dict, Optional, Any, Tuple
from abc import abstractmethod

import torch
from torch import nn

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

class VQLatentPriorModel(nn.Module):
    '''
        Transformer model leanring prior in the latent space of a VQ-VAE
    '''
    def __init__(
            self,
            feature_extractor_model : nn.Module,
            prior_model : Union[PriorModel, Type],
            code_shape : Tuple[int, int],
            **kwargs
        ):
        super().__init__()
        for p in feature_extractor_model.parameters():
            p.requires_grad = False

        self.feature_extractor_model = feature_extractor_model
        self.feature_extractor_model.eval()

        if isinstance(prior_model, PriorModel):
            if prior_model.in_channels != feature_extractor_model.code_size or \
                prior_model.code_shape != tuple(code_shape):
                raise ValueError('Incompatible prior model')
            self.prior_model = prior_model
        elif isinstance(prior_model, type):
            self.prior_model = prior_model(
                in_channels = feature_extractor_model.code_size,
                code_shape = code_shape,
                **kwargs
            )
        else:
            raise ValueError('"prior_model" must be nn.Module or Type')

    def to(self, device, *args, **kwargs):
        '''Override to() to ensure encoder to use correct device'''
        self.feature_extractor_model.to(device)
        self.prior_model.to(device)
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

    def forward(self, x, cond = None):
        '''Retrieve codes for images'''
        code = self.retrieve_codes(x, cond)
        return self.prior_model.forward(code, cond)

    def forward_latent(self, code, cond = None):
        '''Forward pass in latent space'''
        return self.prior_model.forward(code, cond)

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
            codes : torch.Tensor,
            logit_threshold : float,
            num_reconstructions : int = 1,
            is_thres_quantile : bool = False,
            n_iters : int = 1
        ) -> torch.Tensor:
        '''
            Restore from codes by removing unlikely codes
        '''

    def restore_by_codes(
        self,
        codes : torch.Tensor,
        logit_threshold : float,
        num_reconstructions : int = 1,
        is_thres_quantile : bool = False,
        n_iters : int = 1
    ) -> torch.Tensor:
        ''' Restore by embedding codes'''
        return self._restore_abnormal(
            codes, logit_threshold,
            num_reconstructions = num_reconstructions,
            is_thres_quantile = is_thres_quantile,
            n_iters = n_iters
        )

    def restore_images(
        self,
        images : torch.Tensor,
        logit_threshold : float,
        num_reconstructions : int = 1,
        is_thres_quantile : bool = False,
        n_iters : int = 1
    ) -> torch.Tensor:
        ''' Restore images'''
        codes = self.retrieve_codes(images, None)
        return self._restore_abnormal(
            codes, logit_threshold,
            num_reconstructions = num_reconstructions,
            is_thres_quantile = is_thres_quantile,
            n_iters = n_iters
        )

    def _batch_cond(
        self,
        num_images : int,
        cond : Optional[torch.Tensor] = None,
        expected_dim : int = 3
    ) -> torch.Tensor:
        ''' make cond's batch size compatible '''
        if cond is None:
            return None

        if len(cond.shape) == expected_dim - 1:
            return cond.unsqueeze(0).repeat(
                num_images, *([1] * (expected_dim-1))
            )

        assert len(cond.shape) == expected_dim, 'Invalid conditional shape'
        if cond.size(0) == 1:
            return cond.repeat(num_images, *cond.shape[1:])
        if cond.size(0) == num_images:
            return cond

        raise ValueError('Invalid conditional shape')

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
