''' Utility functions for training the prior model '''

from typing import Tuple, Union, Dict, List, Optional, Any
import random

import numpy as np
import torch
import torch.nn.functional as F

from .transformer import PriorTransformer
from ..vqvae import AbstractQuantVAE

def to_onehot(codes : torch.Tensor, code_size : int) -> torch.Tensor:
    ''' convert codes to one-hot tensors '''
    return F.one_hot( # pylint: disable=not-callable
        codes,
        num_classes = code_size
    ).float()

def generate_mask(
        x : torch.Tensor,
        p_mask : Union[float, Tuple[float, float]],
        code_size : int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    ''' generate random mask for training '''
    r_mask = torch.rand(x.shape).to(x.device)
    _p_mask = p_mask if isinstance(p_mask, float) else \
        random.uniform(p_mask[0], p_mask[1]) # type: ignore
    mask = r_mask < _p_mask
    mask.requires_grad = False
    len_mask = mask.sum().item()
    x[mask] = torch.randint(
        low = 0, high = code_size,
        size = (len_mask,) # type: ignore
    ).to(x.device)
    return x, mask

def transformer_loss(
        prior_model : PriorTransformer,
        codes : torch.Tensor,
        p_mask : Union[float, Tuple[float, float]],
        beta : float,
        use_tamper_mask : bool = True,
        reduction : str = 'mean',
        is_training : bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
    ''' get loss for a transformer based prior model '''
    flat_codes = codes.view(codes.size(0), -1)
    code_size = prior_model.in_channels
    inp = to_onehot(flat_codes, code_size)

    if is_training:
        with torch.no_grad():
            flat_clone = flat_codes.clone()
            flat_clone, mask = generate_mask(flat_clone, p_mask, code_size)

        inp_tampered = to_onehot(flat_clone, code_size)
        outp = prior_model.forward(
            inp_tampered,
            mask if use_tamper_mask else None
        )
        loss_t = F.cross_entropy(
            outp[mask], inp[mask], reduction = 'sum', **kwargs
        ) * (1 - beta) + F.cross_entropy(
            outp[~mask], inp[~mask], reduction = 'sum', **kwargs
        ) * beta
        if reduction == 'mean':
            loss_t /= np.prod(codes.shape).tolist()
    else:
        with torch.no_grad():
            outp = prior_model.forward(inp, mask = None)
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

def flatten_codes(codes : torch.Tensor, num_reconstructions : int):
    ''' return flattened codes for reconstruction '''
    return codes.view(codes.size(0), -1).unsqueeze(1).repeat(
            1, num_reconstructions, 1
        ).reshape(codes.size(0) * num_reconstructions, -1)

def resample_codes(
        prior_model : PriorTransformer,
        codes : torch.Tensor,
        logit_threshold : float,
        num_reconstructions : int = 1,
        is_thres_quantile : bool = False,
        n_iters : int = 1,
        use_tamper_mask : bool = True,
        resample_option : str = 'abnormal'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    ''' resample codes for reconstruction '''
    lead_dim = codes.size(0) * num_reconstructions
    code_size = prior_model.in_channels
    with torch.no_grad():
        orig_flat_codes = flatten_codes(codes, num_reconstructions)
        flat_codes = orig_flat_codes.clone()

        for _ in range(n_iters):
            inp = to_onehot(flat_codes, code_size)
            outp = prior_model.forward(inp, None)
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
                    prior_model.forward(inp, err_mask)
                        if use_tamper_mask else outp,
                    dim = -1
                )
                if resample_option == 'all':
                    flat_codes[:] = torch.multinomial(
                        probs.view(-1, code_size), 1
                    ).view(lead_dim, -1)
                elif resample_option == 'abnormal':
                    flat_codes[err_mask] = torch.multinomial(
                        probs[err_mask], 1
                    ).squeeze(-1)
                elif resample_option == 'normal':
                    flat_codes[~err_mask] = torch.multinomial(
                        probs[~err_mask], 1
                    ).squeeze(-1)
                else:
                    raise ValueError('Invalid resample option')
            else:
                break
    return flat_codes, orig_flat_codes

def reconstruct_by_codes(
        feature_extractor_model : AbstractQuantVAE,
        prior_model : Union[PriorTransformer, Tuple[PriorTransformer, PriorTransformer]],
        resampled_flat_codes : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        original_flat_codes : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        num_reconstructions : int = 1,
        **kwargs
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, List[torch.Tensor]]]:
    ''' reconstruct images by codes '''
    with torch.no_grad():
        if isinstance(resampled_flat_codes, torch.Tensor) and \
            isinstance(original_flat_codes, torch.Tensor) and \
            isinstance(prior_model, PriorTransformer):
            code_shape = prior_model.code_shape
            code_size = prior_model.in_channels
            z = feature_extractor_model.embed(
                resampled_flat_codes.view(-1, *code_shape)
            )
            img_recon = torch.stack([
                feature_extractor_model.decode(
                    z[i * num_reconstructions : (i+1) * num_reconstructions]
                ) for i in range(z.size(0) // num_reconstructions)
            ])
            outp = prior_model.forward(
                to_onehot(resampled_flat_codes, code_size),
                mask = None
            )
            losses = F.cross_entropy(
                outp.transpose(-1, 1),
                to_onehot(original_flat_codes, code_size).transpose(-1, 1),
                reduction = 'none',
                **kwargs
            ).mean(dim = -1).view(-1, num_reconstructions)
        else:
            assert isinstance(resampled_flat_codes, tuple) and \
                isinstance(original_flat_codes, tuple) and \
                isinstance(prior_model, tuple)
            z_base, z_detail = feature_extractor_model.embed((
                resampled_flat_codes[0].view(-1, *prior_model[0].code_shape),
                resampled_flat_codes[1].view(-1, *prior_model[1].code_shape)
            ))
            img_recon = torch.stack([
                feature_extractor_model.decode((
                    z_base[i * num_reconstructions : (i+1) * num_reconstructions],
                    z_detail[i * num_reconstructions : (i+1) * num_reconstructions]
                )) for i in range(z_base.size(0) // num_reconstructions)
            ])
            losses = []
            for m_, c_, oc_ in zip(prior_model, resampled_flat_codes, original_flat_codes):
                losses.append(
                    F.cross_entropy(
                        m_.forward(to_onehot(c_, m_.in_channels), mask = None).transpose(-1, 1),
                        to_onehot(oc_, m_.in_channels).transpose(-1, 1),
                        reduction = 'none',
                        **kwargs
                    ).mean(dim = -1).view(-1, num_reconstructions)
                )
            losses = torch.sum(torch.vstack(losses), dim = 0, keepdim = True)
    return img_recon, losses

def batch_cond(
        num_images : int,
        cond : Optional[torch.Tensor],
        expected_dim : int
    ) -> Optional[torch.Tensor]:
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

def create_sample_cond(
        image_chw : Optional[Tuple[int, int, int]],
        cond : Optional[torch.Tensor],
        num_images : int,
        device : Union[str, torch.device],
        kwargs : Dict[str, Any]
    ) -> Tuple[torch.Tensor, int, float, Dict]:
    ''' create conditional image and settings for sampling '''
    if cond is None:
        assert image_chw is not None, 'Require input image shape'
        cond = torch.rand(num_images, *image_chw).to(device)
        n_iters = kwargs.pop('n_iters', 3)
        loss_quantile = kwargs.pop('loss_quantile', 0.1)
    else:
        assert isinstance(cond, torch.Tensor)
        if image_chw is not None:
            assert tuple(cond.shape[-3:]) == tuple(image_chw), \
                'Incompatible conditional shape'
        else:
            image_chw = tuple(cond.shape[-3:])
        cond = batch_cond(num_images, cond, expected_dim = 4).to(device)
        n_iters = kwargs.pop('n_iters', 1)
        loss_quantile = kwargs.pop('loss_quantile', 0.75)
    return cond, n_iters, loss_quantile, kwargs
