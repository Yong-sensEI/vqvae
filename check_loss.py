'''
    given a model and images, perform stats on the loss
'''

from typing import Dict, List, Optional
import os
import sys
from glob import glob
from threading import Event
import argparse
import signal

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

from yw_basics.dataloader import ImageClassificationDataset

from utils import load_model_from_state_dict
from models.prior.base import VQLatentPriorModel

STOP_SIG = Event()

def signal_handler(_, __):
    '''
        Handle the signal to stop execution.
    '''
    print("Received stop signal, stopping running...")
    STOP_SIG.set()

def get_loss(
        model : VQLatentPriorModel,
        model_cfg : Dict,
        image_files : str,
        input_size : List[int],
        max_img_num : Optional[int] = None
    ) -> List[np.ndarray]:
    ''' compute loss from a model file '''
    if os.path.isdir(image_files):
        img_files = glob(os.path.join(image_files, '*.*'))
    else:
        img_files = glob(image_files)

    img_files = [
       os.path.realpath(f) for f in img_files
        if ImageClassificationDataset.is_image_file(f)
    ]

    norm_opt = model_cfg['train']['data'].get('normalization', 'default')
    colorspace = model_cfg['train']['data'].get('colorspace', None)

    trans_cfg = [
        {
            "type": "torchvision.transforms.v2.Resize",
            "args": [input_size,]
        }
    ]

    dat_set = ImageClassificationDataset(
        label_files = [],
        image_files = img_files,
        transforms_configs = trans_cfg,
        normalization_option = norm_opt,
        colorspace = colorspace
    )
    print(f'{dat_set.total_num_samples} images found for evaluation.')
    if max_img_num is None:
        max_img_num = dat_set.total_num_samples

    dat_loader : DataLoader = DataLoader(
        dat_set,
        batch_size = 1,
        shuffle = False
    )

    losses = []
    STOP_SIG.clear()

    for i_, batch in enumerate(tqdm(dat_loader)):
        if STOP_SIG.is_set() or i_ > max_img_num:
            break

        batch_dev = batch.to(next(model.parameters()).device)
        with torch.no_grad():
            losses.append(
                model.loss(
                    batch_dev,
                    reduction = 'none',
                    is_training = False
                )['loss'][0].detach().cpu().numpy()
            )
    return losses

def main():
    ''' main function to run the evaluation '''
    parser = argparse.ArgumentParser(
        description="Compute loss statistics for a VQ latent prior model."
    )
    parser.add_argument(
        "image_files", type=str,
        help="Path to the image files or directory to evaluate on."
    )
    parser.add_argument(
        "-m", "--model-file", type=str, required=True,
        help="Path to the model file to evaluate."
    )
    parser.add_argument(
        "--img-size", type=int, required=True,
        help="Input size (height==width) for the model."
    )
    parser.add_argument(
        "--device", type=str, required=False, default='cuda',
        help="Device to run the evaluation on (default: 'cuda')."
    )
    parser.add_argument(
        "--max-img-num", type=int, required=False, default=None,
        help="Maximum number of images to evaluate (default: None, meaning all)."
    )

    if len(sys.argv) == 1:
        sys.argv.append("-h")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)

    model, model_cfg = load_model_from_state_dict(
        torch.load(args.model_file, weights_only=False), None
    )
    model.to(args.device)
    losses = get_loss(
        model, model_cfg, args.image_files,
        [args.img_size,]*2,
        args.max_img_num
    )

    if len(losses) == 0:
        return

    print(f"Loss statistics for {len(losses)} images:")
    print(f"Mean: {np.mean(losses):.4f}")
    print(f"Min: {np.min(losses):.4f}")
    print(f"10% percentile: {np.percentile(losses, 10):.4f}")
    print(f"30% percentile: {np.percentile(losses, 30):.4f}")
    print(f"Median: {np.median(losses):.4f}")
    print(f"70% percentile: {np.percentile(losses, 70):.4f}")
    print(f"90% percentile: {np.percentile(losses, 90):.4f}")
    print(f"Max: {np.max(losses):.4f}")

if __name__ == "__main__":
    main()
