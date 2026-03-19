''' utils functions '''

from typing import Tuple
import random
import torch

def mask_by_patches(
        x : torch.Tensor,
        n_patches : Tuple[int, int],
        patch_prob : float
    ):
    '''
    Mask input images by patches
    x: (B, C, H, W)
    '''
    if (n_patches[0] <= 1 and n_patches[1] <= 1) or \
        random.uniform(0, 1) < patch_prob:
        return x

    n_b = int(random.uniform(0, 1) < 0.5)

    _, _, height, width = x.shape
    patch_y = height // n_patches[0]
    patch_x = width // n_patches[1]
    mask = torch.zeros_like(x)
    for i in range(n_patches[0]):
        for j in range(n_patches[1]):
            if (i + j) % 2 == n_b:
                mask[:, :, i*patch_y:(i+1)*patch_y, j*patch_x:(j+1)*patch_x] = 1
    return x * mask
