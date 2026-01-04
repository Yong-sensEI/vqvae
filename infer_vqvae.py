'''
    Infer module for VQ-VAE model.
'''

import os
import sys
import argparse
from threading import Event
import signal
import glob
import json

from tqdm import tqdm
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from yw_basics.dataloader import ImageClassificationDataset
from yw_basics.utils import current_datetime

from utils import load_model_from_state_dict

STOP_SIG = Event()

def signal_handler(_, __):
    '''
        Handle the signal to stop training.
    '''
    print("Received stop signal, stopping training...")
    STOP_SIG.set()

def parse_args():
    ''' Parse command line arguments '''
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser(description = "Evaluate a classifier model.")
    parser.add_argument(
        "-m", "--model",
        type=str, required=True,
        help="Path to the model file."
    )
    parser.add_argument(
        "-d", "--data",
        type=str, required=True,
        help="Path to the dataset file."
    )
    parser.add_argument(
        "--img-size",
        type=int, required=True,
        help="Input image size for the model."
    )
    parser.add_argument(
        "-o", "--output",
        type=str, required=False, default='./reconstructions',
        help="Directory to save reconstructed images."
    )
    parser.add_argument(
        "--device",
        type=str, required=False, default='cuda',
        help="Device to run the evaluation on (default: 'cuda')."
    )
    parser.add_argument(
        "--normalize",
        type=str, required=False, default='default',
        help="Normalization method to use (default: 'default')."
    )
    parser.add_argument(
        "--cfg",
        type=str, required=False,
        help="Configuration file for the model."
    )

    if len(sys.argv) == 1:
        sys.argv.append("-h")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model weights {args.model} does not exist.")
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data file {args.data} does not exist.")

    return parser.parse_args()

def eval_model(
        args : argparse.Namespace
    ):
    '''
        evaluate model on the given dataset
    '''
    print(f"Loading model from {args.model}")

    if args.cfg is None:
        cfg = None
    else:
        with open(args.cfg, 'r', encoding='utf-8') as f_:
            cfg = json.load(f_)

    try:
        model, cfg = load_model_from_state_dict(
            torch.load(args.model, weights_only=False),
            'vqvae.VQVAE',
            cfg
        )
    except TypeError as e:
        print(f"Error loading model: {e}. Configuration missing?")
        return

    model.eval()
    dev = torch.device(args.device)
    model.to(dev)

    if os.path.exists(args.output) and not os.path.isdir(args.output):
        raise NotADirectoryError(
            f"Output path {args.output} is not a directory."
        )
    os.makedirs(args.output, exist_ok=True)

    if os.path.isdir(args.data):
        img_files = glob.glob(os.path.join(args.data, '*.*'))
        label_files = []
    elif ImageClassificationDataset.is_image_file(args.data):
        img_files = [args.data]
        label_files = []
    else:
        img_files = None
        label_files = args.data

    dat_set = ImageClassificationDataset(
        label_files=label_files,
        image_files=img_files,
        transforms_configs=[
            {
                "type": "torchvision.transforms.v2.Resize",
                "args": [
                    (args.img_size, args.img_size)
                ]
            }
        ],
        normalization_option=args.normalize,
        colorspace=cfg['train']['data'].get('colorspace', None)
    )
    print(f'Loaded {len(dat_set)} images for inference.')

    dat_loader : DataLoader = DataLoader(
        dat_set,
        batch_size=1,
        shuffle=False
    )

    img_files = [
        (f_, os.path.splitext(os.path.basename(f_))[0])
        for f_ in dat_set.image_files
    ]
    loss_lines = []

    def cvt_color(img):
        ''' Convert color space of an image '''
        if not isinstance(dat_set.colorspace, str):
            return img

        flag = getattr(
            cv2,
            f'COLOR_RGB2{dat_set.colorspace.upper()}',
            None
        )

        if flag is None:
            return img

        return cv2.cvtColor(img, flag)

    for i_, batch in enumerate(tqdm(dat_loader)):
        if STOP_SIG.is_set():
            break

        full_f, img_f = img_files[i_]
        orig_size = dat_set.get_image_size(full_f)

        with torch.no_grad():
            embd_ls, x_hat, perplexity, _ = model(batch.to(dev))

        img = dat_set.image_tensor_to_numpy(x_hat[0].cpu())
        orig_img = dat_set.image_tensor_to_numpy(batch[0].cpu())
        if orig_size is not None:
            img = cv2.resize(img, orig_size)
            orig_img = cv2.resize(orig_img, orig_size)
        cv2.imwrite(
            os.path.join(args.output, img_f + '.jpg'),
            cvt_color(img)
        )

        recon_ls = np.mean((img - orig_img) ** 2)
        loss_lines.append(
            ','.join([
                img_f,
                f"{embd_ls.item():.6f}",
                f"{perplexity.item():.6f}",
                f"{recon_ls:.6f}"
            ])
        )

    with open(os.path.join(
            args.output,
            f'losses_{current_datetime()}.csv'
        ), 'w', encoding='utf-8') as f_:
        f_.write("image_file, embedding_loss, perplexity, recon_loss\n")
        f_.write("\n".join(loss_lines))

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    try:
        eval_model(parse_args())
    except NotADirectoryError as e:
        print(e)
        sys.exit(1)
