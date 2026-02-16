''' VQ-VAE Main Training Script '''

import argparse
import threading
import json
import signal
import copy

from tqdm import tqdm
import numpy as np
import torch

from yw_basics.utils import import_object

import utils

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
    parser.add_argument("-c", "--cfg", type=str, required=True)
    return parser.parse_args()

def train(args):
    ''' Train VQ-VAE model '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.cfg, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    orig_cfg = copy.deepcopy(cfg)
    train_cfg = cfg['train']
    model_cfg = cfg['model']
    data_cfg = train_cfg['data']

    if 'encoder' not in cfg:
        raise ValueError("Encoder configuration must be provided in the config.")

    encoder_cfg = cfg['encoder']

    if 'checkpoint' not in encoder_cfg:
        raise ValueError("Encoder checkpoint path must be provided in the config.")

    ckpt = torch.load(
        encoder_cfg['checkpoint'],
        weights_only = False
    )
    encoder = utils.load_model_from_state_dict(ckpt, 'vqvae.VQVAE', None)[0]
    encoder.to(device)
    encoder.eval()

    # Load data and define batch data loaders
    training_data, validation_data = utils.get_datasets(data_cfg)
    training_loader, validation_loader = utils.get_data_loaders(
        training_data, validation_data, data_cfg
    )

    model_type = import_object(model_cfg.pop('type'))
    model = model_type(
        feature_extractor_model = encoder,
        **model_cfg
    )
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
            'loss': [],
            'accuracy': []
        }

        model.train()
        encoder.eval()
        for _ in tqdm(range(train_cfg['num_steps_train'])):
            if STOP_SIG.is_set():
                break

            x = next(iter(training_loader))
            x = x.to(device)
            optimizer.zero_grad()

            loss_pred = model.loss(x)
            loss = loss_pred['loss']
            acc = (loss_pred['pred'] == loss_pred['target']).float().mean()

            loss.backward()
            optimizer.step()

            results["loss"].append(loss.item())
            results["accuracy"].append(acc.item())

        scheduler.step()

        if not STOP_SIG.is_set():
            print(
                f"Training loss: {np.mean(results['loss']):.4f}",
                f"accuracy: {np.mean(results['accuracy']):.4f}"
            )
        return results

    def validate_epoch():
        results = {
            'loss': [],
            'accuracy': []
        }

        model.eval()
        encoder.eval()
        for _ in tqdm(range(train_cfg['num_steps_validation'])):
            if STOP_SIG.is_set():
                break

            x = next(iter(validation_loader))
            x = x.to(device)

            with torch.no_grad():
                loss_pred = model.loss(x)
            loss = loss_pred['loss']
            acc = (loss_pred['pred'] == loss_pred['target']).float().mean()

            results["loss"].append(loss.item())
            results["accuracy"].append(acc.item())

        if not STOP_SIG.is_set():
            print(
                f"Validation loss: {np.mean(results['loss']):.4f}",
                f"accuracy: {np.mean(results['accuracy']):.4f}"
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
                ckpt_cfg.get('keyword', 'pixelsnail') + ('_final' if is_final else '')
            )
            print(f"Model saved at epoch {epoch + 1}")

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    train(parse_args())
