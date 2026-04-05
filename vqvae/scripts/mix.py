'''
Mix multiple conditional images to generate new images
'''

from typing import Optional, List
import os
import sys
import argparse
from glob import glob

import torch
from PIL import Image
import shortuuid

from yw_basics.dataloader import ImageNormalizer, ImageTaskDataset
from vqvae.models.prior.vqvae_transformer import VQLatentTransformer
from vqvae.utils import load_model_from_state_dict, parse_kwargs

def mix_images(
    model : VQLatentTransformer,
    normalizer : ImageNormalizer,
    condition_image_files : List[str],
    output_dir : str,
    num_samples : int = 1,
    device : Optional[torch.device] = None,
    **kwargs
) -> None:
    ''' mix multiple conditional images to generate new images'''
    if len(condition_image_files) == 0:
        return

    cond_imgs = [Image.open(img_f) for img_f in condition_image_files]
    orig_size = cond_imgs[0].size

    dev = device or torch.device('cuda')
    model.to(dev)
    conds = torch.stack(
        [normalizer(img).to(dev) for img in cond_imgs]
    )

    mixture = model.mix_sample(num_samples, conds, **kwargs).cpu()[0]
    imgs = normalizer.tensor_to_image(mixture)

    for i, img in enumerate(imgs):
        img = img.resize(orig_size)
        img.save(
            os.path.join(output_dir,
            f'{shortuuid.random(8)}_{i}.png')
        )

def main():
    ''' main function to parse arguments and run the mixing process '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model', required = True,
        help = 'Prior model file path'
    )
    parser.add_argument(
        '-i', '--condition_images', nargs='+', required = True,
        help = 'List of conditional image file paths'
    )
    parser.add_argument(
        '-o', '--output_dir', required = True,
        help = 'Directory to save generated images'
    )
    parser.add_argument(
        '-n', '--num_samples', type = int, default = 1, 
        help = 'Number of images to generate'
    )
    parser.add_argument(
        "--device", default = 'cuda',
        help = "Device to run the evaluation on (default: 'cuda')."
    )
    parser.add_argument(
        "--kwargs", type=str, nargs='*', default=[],
        help="Device to run the evaluation on (default: 'cuda')."
    )

    if len(sys.argv) == 1:
        sys.argv.append("-h")
    args = parser.parse_args()

    model, cfg = load_model_from_state_dict(args.model, None)
    assert isinstance(model, VQLatentTransformer), 'Invalid model file'
    normalizer : ImageNormalizer = ImageNormalizer(
        cfg['train']['data'].get('transforms', []),
        cfg['train']['data'].get('normalization', None),
        cfg['train']['data'].get('colorspace', None),
        eval_mode = True
    )

    os.makedirs(args.output_dir, exist_ok = True)

    img_files = []
    for img in args.condition_images:
        if os.path.isfile(img):
            img_files.append(img)
        else:
            if os.path.isdir(img):
                img = os.path.join(img, "*.*")
            img_files.extend([
                f for f in glob(img) if ImageTaskDataset.is_image_file(f)
            ])

    kwargs = parse_kwargs(args.kwargs)
    mix_images(
        model,
        normalizer,
        img_files,
        args.output_dir,
        args.num_samples,
        torch.device(args.device),
        **kwargs
    )

if __name__ == '__main__':
    main()
