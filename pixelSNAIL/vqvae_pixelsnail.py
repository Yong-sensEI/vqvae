'''
    PixelSNAIL model in VQ-VAE's latent space
'''

from collections import OrderedDict

import torch
from torch.nn import functional as F

from .pixelsnail import PixelSNAIL

class VQLatentSNAIL(PixelSNAIL):
    '''
        PixelSNAIL model operating in the latent space of a VQ-VAE
    '''
    def __init__(self, feature_extractor_model, **kwargs):
        super().__init__(
            in_channels = feature_extractor_model.code_size,
            **kwargs
        )

        for p in feature_extractor_model.parameters():
            p.requires_grad = False

        self.feature_extractor_model = feature_extractor_model
        self.feature_extractor_model.eval()

    def to(self, device, *args, **kwargs):
        '''Override to() to ensure encoder to use correct device'''
        self.feature_extractor_model.to(device)
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
        return super().forward(code, cond)

    def forward_latent(self, code, cond = None):
        '''Forward pass in latent space'''
        return super().forward(code, cond)

    def loss(self, x, cond = None, reduction = 'mean'):
        '''retrieve codes for images'''
        code = self.retrieve_codes(x, cond)
        logits = super().forward(code, cond)
        nll = F.cross_entropy(logits, code, reduction = reduction)
        return OrderedDict(
            loss = nll,
            pred = torch.argmax(logits, dim=1),
            target = code
        )

    def sample(self, n, img_size = (64,64), cond = None):
        device = next(self.parameters()).device
        samples = torch.zeros(n, *img_size).long().to(device)
        with torch.no_grad():
            for r in range(img_size[0]):
                for c in range(img_size[1]):
                    if self.cond_channels is not None:
                        logits = super().forward(samples,cond)[:, :, r, c]
                    else:
                        logits = super().forward(samples)[:, :, r, c]
                    probs = F.softmax(logits, dim=1)
                    samples[:, r, c] = torch.multinomial(probs, 1).squeeze(-1)
        return samples.cpu().numpy()
