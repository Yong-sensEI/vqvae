'''
    Utility functions for data loading, timestamp generation, and model saving.
'''
from typing import Dict, Optional, Tuple
import os

import torch
from torch.utils.data import DataLoader

from yw_basics.dataloader import ImageClassificationDataset
from yw_basics.utils import current_datetime

from models.vqvae import VQVAE

def get_datasets(data_cfg : Dict):
    '''
        Create torch dataset.
    '''
    train_data = ImageClassificationDataset(
        label_files = data_cfg['train_files'],
        transforms_configs = data_cfg.get('transforms', []),
        normalization_option = data_cfg.get('normalization', None),
        colorspace=data_cfg.get('colorspace', None)
    )
    val_data = ImageClassificationDataset(
        label_files = data_cfg['validation_files'],
        transforms_configs = data_cfg.get('transforms', []),
        normalization_option = data_cfg.get('normalization', None),
        colorspace=data_cfg.get('colorspace', None)
    )
    return train_data, val_data

def get_data_variance(
        dataset : ImageClassificationDataset,
        max_batches : Optional[int] = None
    ):
    '''
        Calculate variance of the dataset.
    '''
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        pin_memory=True
    )
    all_data_tensor = torch.cat(
        list(data_loader)
            if max_batches is None or dataset.num_samples(None) < max_batches else
        [next(iter(data_loader)) for _ in range(max_batches)],
        dim=0
    )
    return torch.var(all_data_tensor).numpy().tolist()

def get_data_loaders(
        train_data : ImageClassificationDataset,
        val_data : ImageClassificationDataset,
        dat_cfg : Dict
    ):
    '''
        Create data loaders for training and validation datasets.
    '''
    train_loader = DataLoader(
        train_data,
        batch_size=dat_cfg['loader']['batch_size'],
        shuffle=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_data,
        batch_size=dat_cfg['loader']['batch_size'],
        shuffle=True,
        pin_memory=True
    )
    return train_loader, val_loader

def save_model_and_results(
        save_path : str,
        model : torch.nn.Module,
        model_cfg : Dict,
        train_results : Dict,
        val_results : Dict
    ):
    '''
        Save the model state and training results to a file.
    '''
    results_to_save = {
        'model': model.state_dict(),
        'config': model_cfg,
        'train_results': train_results,
        'validation_results': val_results
    }
    torch.save(
        results_to_save,
        os.path.join(
            save_path, 'vqvae_data_' + current_datetime() + '.pth'
        )
    )

def load_model_from_state_dict(
        state_dict : Dict,
        config : Optional[Dict] = None
    ) -> Tuple[torch.nn.Module, Dict]:
    '''
        load model from the a .pt file
    '''
    if config is None:
        config = state_dict.get('config', None)

    model_cfg = config['model']

    model = VQVAE(
        model_cfg['num_hidden'],
        model_cfg['num_residual_hidden'],
        model_cfg['residual_layers'],
        model_cfg['num_embeddings'],
        model_cfg['embedding_dim'],
        model_cfg['commitment_cost']
    )

    model.load_state_dict(state_dict['model'], strict=True)

    return model, config
