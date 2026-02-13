''' VQ-VAE Main Training Script '''

import argparse
import threading
import json
import signal

from tqdm import tqdm
import numpy as np
import torch

from yw_basics.utils import import_object

import utils
from vqvae import VQVAE

STOP_SIG = threading.Event()

def signal_handler(_, __):
    '''
        Handle the signal to stop training.
    '''
    print("Received stop signal, stopping training...")
    STOP_SIG.set()

def parse_args():
    ''' Parse command line arguments '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    return parser.parse_args()

def train(args):
    ''' Train VQ-VAE model '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.cfg, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    orig_cfg = cfg.copy()
    train_cfg = cfg['train']
    model_cfg = cfg['model']
    data_cfg = train_cfg['data']

    # Load data and define batch data loaders
    training_data, validation_data = utils.get_datasets(data_cfg)

    loss_type = train_cfg.get('loss_type', 'l2')
    if loss_type == 'l2':
        x_train_var = utils.get_data_variance(training_data, 1000)
        x_eval_var = utils.get_data_variance(validation_data, 1000)
        print(f'Training data variance: {x_train_var:.4f}')
        print(f'Validation data variance: {x_eval_var:.4f}')
    else:
        x_train_var, x_eval_var = None, None

    training_loader, validation_loader = utils.get_data_loaders(
        training_data, validation_data, data_cfg
    )

    if 'checkpoint' in model_cfg:
        ckpt = torch.load(
            model_cfg['checkpoint'],
            weights_only = False
        )
        model, _ = utils.load_model_from_state_dict(ckpt, 'vqvae.VQVAE', None)
    else:
        model = VQVAE(**model_cfg)
    model.to(device)

    # Set up optimizer and training loop
    optimizer = import_object(train_cfg['optimizer']['type'])(
        model.parameters(),
        **train_cfg['optimizer'].get('kwargs', {})
    )
    scheduler = import_object(train_cfg['scheduler']['type'])(
        optimizer,
        **train_cfg['scheduler'].get('kwargs', {})
    )

    log_interval = train_cfg.get('checkpoint', {}).get('interval', 100)
    save_path = train_cfg.get('checkpoint', {}).get('path', './results')

    def train_epoch():
        results = {
            'recon_errors': [],
            'embedding_loss': [],
            'perplexities': [],
        }

        model.train()
        for _ in tqdm(range(train_cfg['num_steps_train'])):
            if STOP_SIG.is_set():
                break

            x = next(iter(training_loader))
            x = x.to(device)
            optimizer.zero_grad()

            embedding_loss, x_hat, perplexity, _ = model(x)
            recon_loss = torch.mean((x_hat - x)**2) / x_train_var \
                if loss_type == 'l2' else torch.mean(torch.abs(x_hat - x))
            loss = recon_loss + embedding_loss

            loss.backward()
            optimizer.step()

            results["recon_errors"].append(recon_loss.item())
            results["perplexities"].append(perplexity.item())
            results["embedding_loss"].append(embedding_loss.item())

        scheduler.step()

        if not STOP_SIG.is_set():
            print(
                f"Training reconstruction error: {np.mean(results['recon_errors']):.4f}",
                f"embedding loss: {np.mean(results['embedding_loss']):.4f}",
                f"perplexity: {np.mean(results['perplexities']):.4f}"
            )
        return results

    def validate_epoch():
        results = {
            'recon_errors': [],
            'embedding_loss': [],
            'perplexities': [],
        }

        model.eval()
        for _ in tqdm(range(train_cfg['num_steps_validation'])):
            if STOP_SIG.is_set():
                break

            x = next(iter(validation_loader))
            x = x.to(device)

            with torch.no_grad():
                embedding_loss, x_hat, perplexity, _ = model(x)
            recon_loss = torch.mean((x_hat - x)**2) / x_eval_var \
                if loss_type == 'l2' else torch.mean(torch.abs(x_hat - x))

            results["recon_errors"].append(recon_loss.item())
            results["perplexities"].append(perplexity.item())
            results["embedding_loss"].append(embedding_loss.item())

        if not STOP_SIG.is_set():
            print(
                f"Validation reconstruction error: {np.mean(results['recon_errors']):.4f}",
                f"embedding loss: {np.mean(results['embedding_loss']):.4f}",
                f"perplexity: {np.mean(results['perplexities']):.4f}"
            )
        return results

    STOP_SIG.clear()

    for epoch in range(train_cfg['epochs']):
        print(f"Epoch {epoch + 1}/{train_cfg['epochs']}")

        train_results = train_epoch()
        if STOP_SIG.is_set():
            break

        val_results = validate_epoch()
        if STOP_SIG.is_set():
            break

        is_final = epoch + 1 == train_cfg['epochs']
        if 'checkpoint' in train_cfg:
            ckpt_cfg = train_cfg['checkpoint']
            if (epoch + 1) % log_interval != 0 and not is_final:
                continue

            utils.save_model_and_results(
                save_path, model, orig_cfg, train_results, val_results,
                ckpt_cfg.get('keyword', 'vqvae') + ('_final' if is_final else '')
            )
            print(f"Model saved at epoch {epoch + 1}")

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    train(parse_args())
