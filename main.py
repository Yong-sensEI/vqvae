''' VQ-VAE Main Training Script '''

import os
import argparse
import threading
import json
import signal

from tqdm import tqdm
import numpy as np
import torch

from yw_basics.utils import import_object

import utils
from models.vqvae import VQVAE

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
    timestamp = utils.current_datetime()

    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--filename",  type=str, default=timestamp)

    # whether or not to save model
    parser.add_argument("--no-save", action="store_true")

    return parser.parse_args()

def train(args):
    ''' Prepare training environment '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.cfg, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    train_cfg = cfg['train']
    model_cfg = cfg['model']
    data_cfg = train_cfg['data']

    # Load data and define batch data loaders
    training_data, validation_data = utils.get_datasets(data_cfg)

    x_train_var = utils.get_data_variance(training_data, 1000)
    x_eval_var = utils.get_data_variance(validation_data, 1000)
    print(f'Training data variance: {x_train_var:.4f}')
    print(f'Validation data variance: {x_eval_var:.4f}')

    training_loader, validation_loader = utils.get_data_loaders(
        training_data, validation_data, data_cfg
    )

    # Set up VQ-VAE model with components defined in ./models/ folder
    model = VQVAE(
        model_cfg['num_hidden'],
        model_cfg['num_residual_hidden'],
        model_cfg['residual_layers'],
        model_cfg['num_embeddings'],
        model_cfg['embedding_dim'],
        model_cfg['commitment_cost']
    ).to(device)

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

            embedding_loss, x_hat, perplexity = model(x)
            recon_loss = torch.mean((x_hat - x)**2) / x_train_var
            loss = recon_loss + embedding_loss

            loss.backward()
            optimizer.step()

            results["recon_errors"].append(recon_loss.cpu().detach().numpy())
            results["perplexities"].append(perplexity.cpu().detach().numpy())
            results["embedding_loss"].append(embedding_loss.cpu().detach().numpy())

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
                embedding_loss, x_hat, perplexity = model(x)
            recon_loss = torch.mean((x_hat - x)**2) / x_eval_var

            results["recon_errors"].append(recon_loss.cpu().detach().numpy())
            results["perplexities"].append(perplexity.cpu().detach().numpy())
            results["embedding_loss"].append(embedding_loss.cpu().detach().numpy())

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
            if args.no_save or (
                    (epoch + 1) % log_interval != 0 and not is_final
                ):
                continue

            utils.save_model_and_results(
                save_path, model, cfg, train_results, val_results
            )
            print(f"Model saved at epoch {epoch + 1}")

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    train(parse_args())
