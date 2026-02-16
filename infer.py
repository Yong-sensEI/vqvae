'''
    Infer module for VQ-VAE model.
'''

import os
import sys
import argparse
from threading import Event
import signal
import glob

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
        Handle the signal to stop execution.
    '''
    print("Received stop signal, stopping running...")
    STOP_SIG.set()

def parse_args():
    ''' Parse command line arguments '''
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser(description = "Evaluate a classifier model.")
    parser.add_argument(
        "-d", "--data", type=str, required=True,
        help="Path to the dataset file."
    )
    parser.add_argument(
        "--img-size", type=int, required=True,
        help="Input image size for the model."
    )
    parser.add_argument(
        "-vm", "--vae-model", type=str, required=False,
        help="Path to the VAE model file."
    )
    parser.add_argument(
        "-pm", "--prior-model", type=str, required=False,
        help="Path to the prior model file."
    )
    parser.add_argument(
        "-o", "--output", type=str, required=False, default='./reconstructions',
        help="Directory to save reconstructed images."
    )
    parser.add_argument(
        "--device", type=str, required=False, default='cuda',
        help="Device to run the evaluation on (default: 'cuda')."
    )
    parser.add_argument(
        "--normalize", type=str, required=False, default='default',
        help="Normalization method to use (default: 'default')."
    )
    parser.add_argument(
        '--logit-thres', type=float, required=False, default=0.0,
        help='Logit threshold to computer anomaly score'
    )
    parser.add_argument(
        '--save-recon', action='store_true',
        help='Save reconstructed images'
    )
    #parser.add_argument(
    #    '--blur-ker', type=int, default=0,
    #    help='Apply Gaussian blur with the given kernel size before inference'
    #)
    parser.add_argument(
        '--restore-num', type=int, default=0,
        help='Number of restored images'
    )

    if len(sys.argv) == 1:
        sys.argv.append("-h")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data file {args.data} does not exist.")

    args = parser.parse_args()

    if len(args.normalize) == 0 or args.normalize.lower() == 'none' or \
        args.normalize.lower() == 'null':
        args.normalize = None

    return args

def eval_model(
        args : argparse.Namespace
    ):
    '''
        evaluate model on the given dataset
    '''

    vae_model, prior_model = None, None
    vae_cfg, prior_cfg = None, None
    if isinstance(args.vae_model, str):
        print(f"Loading VAE model from {args.vae_model}")
        try:
            vae_model, vae_cfg = load_model_from_state_dict(
                torch.load(args.vae_model, weights_only=False),
                'vqvae.VQVAE', None
            )
        except FileNotFoundError as e:
            print(f"Error loading VAE model: {e}.")
    if isinstance(args.prior_model, str):
        print(f"Loading prior model from {args.prior_model}")
        try:
            prior_model, prior_cfg = load_model_from_state_dict(
                torch.load(args.prior_model, weights_only = False),
                None, None
            )
        except FileNotFoundError as e:
            print(f"Error loading prior model: {e}.")

    if vae_model is None and prior_model is None:
        print('No models provided')
        sys.exit(1)

    if isinstance(vae_cfg, dict):
        colorspace = vae_cfg['train']['data'].get('colorspace', None)
    else:
        colorspace = prior_cfg['train']['data'].get('colorspace', None)

    dev = torch.device(args.device)
    if vae_model is not None:
        vae_model.eval()
        vae_model.to(dev)
    if prior_model is not None:
        prior_model.eval()
        prior_model.to(dev)
        if vae_model is None:
            vae_model = prior_model.feature_extractor_model

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

    trans_cfg = [
        {
            "type": "torchvision.transforms.v2.Resize",
            "args": [
                (args.img_size, args.img_size)
            ]
        }
    ]
    #if args.blur_ker > 0:
    #    trans_cfg.append(
    #        {
    #            "type": "torchvision.transforms.v2.GaussianBlur",
    #            "kwargs": {
    #                "kernel_size": args.blur_ker
    #            }
    #        }
    #    )
    dat_set = ImageClassificationDataset(
        label_files = label_files,
        image_files = img_files,
        transforms_configs = trans_cfg,
        normalization_option = args.normalize,
        colorspace = colorspace
    )
    print(f'Loaded {len(dat_set)} images for inference.')

    dat_loader : DataLoader = DataLoader(
        dat_set,
        batch_size = 1,
        shuffle = False
    )

    img_files = [
        (f_, os.path.splitext(os.path.basename(f_))[0])
        for f_ in dat_set.image_files
    ]
    losses = {
        'embedding_loss': [],
        'perplexity': [],
        'recon_loss': [],
        'anomaly_score': []
    }

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

    STOP_SIG.clear()
    stop_i = 0

    for i_, batch in enumerate(tqdm(dat_loader)):
        if STOP_SIG.is_set():
            break

        full_f, img_f = img_files[i_]
        orig_size = dat_set.get_image_size(full_f)
        batch_dev = batch.to(dev)

        if vae_model is not None:
            with torch.no_grad():
                embd_ls, x_hat, perplexity, _ = vae_model(batch_dev)

            img = dat_set.image_tensor_to_numpy(x_hat[0].cpu())
            orig_img = dat_set.image_tensor_to_numpy(batch[0])
            if orig_size is not None:
                img = cv2.resize(img, orig_size)
                orig_img = cv2.resize(orig_img, orig_size)
            if args.save_recon or args.restore_num > 0:
                cv2.imwrite(
                    os.path.join(args.output, img_f + '.jpg'),
                    cvt_color(img)
                )

            losses['recon_loss'].append(
                np.sqrt(np.mean((img - orig_img) ** 2))
            )
            losses['embedding_loss'].append(embd_ls.item())
            losses['perplexity'].append(perplexity.item())
        else:
            losses['recon_loss'].append(None)
            losses['embedding_loss'].append(None)
            losses['perplexity'].append(None)

        if prior_model is not None:
            with torch.no_grad():
                loss_dict = prior_model.loss(
                    batch_dev, reduction='none', is_training = False
                )

            loss = loss_dict['loss'][0]
            score = torch.sum(
                loss * (loss > args.logit_thres)
            ).item()
            losses['anomaly_score'].append(score)

            if args.restore_num > 0:
                restores = prior_model.restore_by_codes(
                    loss_dict['target'],
                    args.logit_thres,
                    args.restore_num
                )
                for i, res_img in enumerate(restores):
                    img = dat_set.image_tensor_to_numpy(res_img.cpu())
                    if orig_size is not None:
                        img = cv2.resize(img, orig_size)
                    cv2.imwrite(
                        os.path.join(args.output, img_f + f'_restore_{i}.jpg'),
                        cvt_color(img)
                    )
        else:
            losses['anomaly_score'].append(None)

        stop_i = i_

    with open(os.path.join(
            args.output,
            f'losses_{current_datetime()}.csv'
        ), 'w', encoding='utf-8') as f_:
        f_.write("image_file," + ','.join(losses.keys()) + "\n")

        for i_, img_f in enumerate(img_files):
            if i_ > stop_i:
                break

            f_.write(img_f[1] + ',' + ','.join([
                    '' if val[i_] is None else f'{val[i_]:.6f}'
                    for val in losses.values()
                ]) + '\n'
            )

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    try:
        eval_model(parse_args())
    except NotADirectoryError as e:
        print(e)
        sys.exit(1)
