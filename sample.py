'''
    Generate synthetic images
'''
from typing import Optional, Tuple
import argparse
import sys
import os
from glob import glob
import threading
import signal

from tqdm import tqdm
import shortuuid
from PIL import Image
import torch

from yw_basics.dataloader import ImageNormalizer, ImageTaskDataset

from utils import load_model_from_state_dict, parse_kwargs
from models.prior.base import AbstractVQLatentPriorModel

STOP_SIG = threading.Event()

def signal_handler(_, __):
    '''
        Handle the signal to stop execution.
    '''
    print("Received stop signal, stopping running...")
    STOP_SIG.set()

def gen_images(
    model : AbstractVQLatentPriorModel,
    normalizer : ImageNormalizer,
    output_dir : str,
    image_chw : Optional[Tuple[int, int, int]] = None,
    condition_image_file : Optional[str] = None,
    num_samples : int = 1,
    device : Optional[torch.device] = None,
    **kwargs
) -> None:
    '''
    Generate images, potentially based on a conditional image
    '''
    condition_image : Optional[Image.Image] = None
    if isinstance(condition_image_file, str):
        condition_image = Image.open(condition_image_file)
        img_name = os.path.splitext(os.path.basename(condition_image_file))[0]
        orig_size = condition_image.size
    else:
        img_name = shortuuid.ShortUUID().random(8)
        orig_size = None

    cond = None if condition_image is None else normalizer(condition_image)

    dev = device or torch.device('cuda')
    model.to(dev)
    if cond is not None:
        cond.to(dev)

    img_tensors = model.sample(num_samples, cond, image_chw, **kwargs).detach().cpu()[0]
    imgs = normalizer.tensor_to_image(img_tensors)

    if isinstance(imgs, Image.Image):
        imgs.save(os.path.join(output_dir, img_name + '.png'))
    elif len(imgs) == 1:
        img = imgs[0]
        if orig_size is not None:
            img = img.resize(orig_size)
        img.save(os.path.join(output_dir, img_name + '.png'))
    else:
        for i, img in enumerate(imgs):
            if orig_size is not None:
                img = img.resize(orig_size)
            img.save(os.path.join(output_dir, img_name + f'_{i}.png'))

def parse_args():
    ''' Parse command line arguments '''
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m", "--model", type=str, required=True,
        help="Path to the prior model file."
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, required=True,
        help="Path to save generated images."
    )
    parser.add_argument(
        "--input-chw", type=int, nargs=3, required=False,
        help="Input image C x H x W"
    )
    parser.add_argument(
        "--device", type=str, default='cuda',
        help="Device to run the evaluation on (default: 'cuda')."
    )
    parser.add_argument(
        "--kwargs", type=str, nargs='*', default=[],
        help="Device to run the evaluation on (default: 'cuda')."
    )
    parser.add_argument(
        "-i", "--images", type=str, required=False,
        help="Conditional images"
    )
    parser.add_argument(
        "-n", "--num-images", type=int, default=1,
        help="Number of unconditional images to be generated"
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, default=1,
        help="Number of samples per image"
    )

    if len(sys.argv) == 1:
        sys.argv.append("-h")

    args = parser.parse_args()

    if args.images is None:
        assert args.input_chw is not None, \
            'Input image size is required without input images'

    return args

def main():
    ''' main function '''
    args = parse_args()
    dev = torch.device(args.device)

    if args.images is None or os.path.isfile(args.images):
        img_files = [args.images] * args.num_images
    else:
        if os.path.isdir(args.images):
            args.images = os.path.join(args.images, "*.*")
        img_files = [f for f in glob(args.images) if ImageTaskDataset.is_image_file(f)]

    os.makedirs(args.output_dir, exist_ok = True)

    model, cfg = load_model_from_state_dict(args.model, None)
    assert isinstance(model, AbstractVQLatentPriorModel), 'Invalid model file'
    normalizer : ImageNormalizer = ImageNormalizer(
        cfg['train']['data'].get('transforms', []),
        cfg['train']['data'].get('normalization', None),
        cfg['train']['data'].get('colorspace', None),
        eval_mode = True
    )

    signal.signal(signal.SIGINT, signal_handler)
    kwargs = parse_kwargs(args.kwargs)
    for img_f in tqdm(img_files):
        if STOP_SIG.is_set():
            break

        gen_images(
            model, normalizer, args.output_dir, args.input_chw,
            img_f, args.batch_size, dev, **kwargs
        )

if __name__ == '__main__':
    main()
