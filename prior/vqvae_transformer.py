'''
    Transformer to learn VQ-VAE's latent prior distribution
'''

from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .base import VQLatentPriorModel
from .transformer import PriorTransformer

class VQLatentTransformer(VQLatentPriorModel):
    '''
        Transformer model leanring prior in the latent space of a VQ-VAE
    '''
    def __init__(self, feature_extractor_model : nn.Module, **kwargs):
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
        return F.one_hot( # pylint: disable=not-callable
            codes,
            num_classes = self.feature_extractor_model.code_size
        ).float()

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

        flat_codes = codes.view(x.size(0), -1)
        inp = self.to_onehot(flat_codes)

        if is_training:
            with torch.no_grad():
                flat_clone = flat_codes.clone()
                flat_clone, mask = self.generate_mask(flat_clone)

            inp_tampered = self.to_onehot(flat_clone)
            outp = self.prior_model.forward(
                inp_tampered,
                mask if self._use_tamper_mask else None
            )
            loss_t = F.cross_entropy(
                outp[mask], inp[mask], reduction = 'sum', **kwargs
            ) * (1 - self._beta) + F.cross_entropy(
                outp[~mask], inp[~mask], reduction = 'sum', **kwargs
            ) * self._beta
            if reduction == 'mean':
                loss_t /= np.prod(codes.shape)
        else:
            with torch.no_grad():
                outp = self.prior_model.forward(inp, mask = None)
                loss_t = F.cross_entropy(
                    outp.transpose(-1, 1),
                    inp.transpose(-1, 1),
                    reduction = reduction,
                    **kwargs
                )

        logits = outp.view(*codes.shape, -1)

        return {
            'loss': loss_t,
            'logits': logits,
            'pred': torch.argmax(logits, dim = -1),
            'target': codes
        }

    def generate_mask(self, x : torch.Tensor):
        ''' generate random mask for training '''
        r_mask = torch.rand(x.shape).to(x.device)
        mask = r_mask < self._p_mask
        mask.requires_grad = False
        x[mask] = torch.randint(
            low = 0,
            high = self.feature_extractor_model.code_size,
            size = (torch.sum(mask).item(),)
        ).to(x.device)
        return x, mask

    def _restore_abnormal(
            self,
            codes : torch.Tensor,
            logit_threshold : float,
            num_reconstructions : int = 1,
            is_thres_quantile : bool = False,
            n_iters : int = 1
        ) -> torch.Tensor:
        lead_dim = codes.size(0) * num_reconstructions
        with torch.no_grad():
            flat_codes = codes.view(codes.size(0), -1).unsqueeze(1).repeat(
                1, num_reconstructions, 1
            ).reshape(lead_dim, -1)

            for _ in range(n_iters):
                inp = self.to_onehot(flat_codes)
                outp = self.prior_model.forward(
                    inp, None
                )
                loss = F.cross_entropy(
                    outp.transpose(-1, 1),
                    inp.transpose(-1, 1),
                    reduction = 'none'
                )
                _threshold = torch.quantile(loss, logit_threshold) \
                    if is_thres_quantile else logit_threshold
                err_mask = loss > _threshold
                if torch.any(err_mask):
                    probs = F.softmax(
                        self.prior_model.forward(inp, err_mask)
                            if self._use_tamper_mask else outp,
                        dim = -1
                    )
                    flat_codes[err_mask] = torch.multinomial(
                        probs[err_mask], 1
                    ).squeeze(-1)
                else:
                    break

            z = self.feature_extractor_model.vector_quantization.embed(
                flat_codes.view(lead_dim, *codes.shape[1:])
            )
            img_recon = torch.cat([
                self.feature_extractor_model.decoder(
                    z[i * num_reconstructions : (i+1) * num_reconstructions]
                ) for i in range(codes.size(0))
            ])
        return img_recon

    def sample(
            self,
            num_images : int,
            cond : Optional[torch.Tensor] = None,
            image_chw : Optional[Tuple[int, int, int]] = None,
            **kwargs
        ):
        loss_quantile = kwargs.get('loss_quantile', 0.67)

        dev = next(self.parameters()).device
        if cond is None:
            assert image_chw is not None, 'Require input image shape'
            cond = torch.rand(num_images, *image_chw).to(dev)
            n_iters = kwargs.get('n_iters', 3)
        else:
            assert isinstance(cond, torch.Tensor)
            if image_chw is not None:
                assert tuple(cond.shape[-3:]) == tuple(image_chw[::-1]), \
                    'Incompatible conditional shape'
            cond = self._batch_cond(num_images, cond, expected_dim = 4).to(dev)
            n_iters = kwargs.get('n_iters', 1)

        with torch.no_grad():
            codes = self.retrieve_codes(cond, None)

        return self._restore_abnormal(
            codes,
            logit_threshold = loss_quantile,
            num_reconstructions = 1,
            is_thres_quantile = True,
            n_iters = n_iters
        )
