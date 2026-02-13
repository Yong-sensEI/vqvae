'''
    PixelSNAIL model in VQ-VAE's latent space
'''
from typing import Optional
from collections import OrderedDict

import torch
from torch.nn import functional as F

from .pixelsnail import PixelSNAIL
from .base import VQLatentPriorModel

class VQLatentSNAIL(VQLatentPriorModel):
    '''
        PixelSNAIL model operating in the latent space of a VQ-VAE
    '''
    def __init__(self, feature_extractor_model, **kwargs):
        super().__init__(
            feature_extractor_model = feature_extractor_model,
            prior_model = PixelSNAIL,
            **kwargs
        )

    def loss(
            self,
            x : torch.Tensor,
            reduction : str = 'mean',
            is_training : bool = True,
            **kwargs
        ):
        ''' cross-entropy loss '''
        cond = kwargs.get('cond', None)
        code = self.retrieve_codes(x, cond)
        logits = self.prior_model.forward(code, cond)
        nll = F.cross_entropy(logits, code, reduction = reduction)
        return OrderedDict(
            loss = nll,
            logits = logits,
            pred = torch.argmax(logits, dim=1),
            target = code
        )

    def sample(self, n, img_size = (64,64), cond = None):
        ''' sample an image '''
        device = next(self.parameters()).device
        samples = torch.zeros(n, *img_size).long().to(device)
        with torch.no_grad():
            for r in range(img_size[0]):
                for c in range(img_size[1]):
                    if self.prior_model.cond_channels is not None:
                        logits = self.prior_model.forward(samples,cond)[:, :, r, c]
                    else:
                        logits = self.prior_model.forward(samples)[:, :, r, c]
                    probs = F.softmax(logits, dim=1)
                    samples[:, r, c] = torch.multinomial(probs, 1).squeeze(-1)
        return samples.cpu().numpy()

    def _restore_abnormal(
            self,
            codes : torch.Tensor,
            logit_threshold : float,
            num_reconstructions : int = 1
        ) -> torch.Tensor:
        code_size = codes.shape[-2:]

        with torch.no_grad():
            samples = codes.clone().unsqueeze(1).repeat(
                1, num_reconstructions, 1, 1
            ).reshape(
                codes.size(0) * num_reconstructions, *code_size
            )
            for r in range(code_size[0]):
                for c in range(code_size[1]):
                    logits = self.forward_latent(samples, None)[:, :, r, c]
                    loss = F.cross_entropy(logits, samples[:, r, c], reduction='none')

                    # Replace sample if above threshold
                    probs = F.softmax(logits, dim=1)
                    samples[loss > logit_threshold, r, c] = torch.multinomial(
                        probs, 1).squeeze(-1)[loss > logit_threshold]

            z = self.feature_extractor_model.vector_quantization.embed(samples)
            img_recon = torch.cat([
                self.feature_extractor_model.decoder(
                    z[i * num_reconstructions : (i+1) * num_reconstructions]
                ) for i in range(codes.size(0))
            ])

        return img_recon
