'''
    VQ-VAE's latent prior distribution
'''

from typing import Union, Type, Dict
from abc import abstractmethod

import torch
from torch import nn

class VQLatentPriorModel(nn.Module):
    '''
        Transformer model leanring prior in the latent space of a VQ-VAE
    '''
    def __init__(
            self,
            feature_extractor_model : nn.Module,
            prior_model : Union[nn.Module, Type],
            **kwargs
        ):
        super().__init__()
        for p in feature_extractor_model.parameters():
            p.requires_grad = False

        self.feature_extractor_model = feature_extractor_model
        self.feature_extractor_model.eval()

        if isinstance(prior_model, nn.Module):
            self.prior_model = prior_model
        elif isinstance(prior_model, type):
            self.prior_model = prior_model(
                in_channels = feature_extractor_model.code_size,
                **kwargs
            )
        else:
            raise ValueError('"prior_model" must be nn.Module or Type')

    def to(self, device, *args, **kwargs):
        '''Override to() to ensure encoder to use correct device'''
        self.feature_extractor_model.to(device)
        self.prior_model.to(device)
        super().to(device, *args, **kwargs)

    def retrieve_codes(self,x,cond):
        '''Retrieve discrete latent codes from VQ-VAE'''
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
            num_reconstructions : int = 1
        ) -> torch.Tensor:
        '''
            Restore from codes by removing unlikely codes
        '''

    def restore_by_codes(
        self,
        codes : torch.Tensor,
        logit_threshold : float,
        num_reconstructions : int = 1
    ) -> torch.Tensor:
        ''' Restore by embedding codes'''
        return self._restore_abnormal(
            codes, logit_threshold,
            num_reconstructions = num_reconstructions
        )

    def restore_images(
        self,
        images : torch.Tensor,
        logit_threshold : float,
        num_reconstructions : int = 1
    ) -> torch.Tensor:
        ''' Restore images'''
        codes = self.retrieve_codes(images, None)
        return self._restore_abnormal(
            codes, logit_threshold,
            num_reconstructions = num_reconstructions
        )
