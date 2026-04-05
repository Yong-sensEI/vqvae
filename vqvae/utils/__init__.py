'''
    Utility functions for data loading, timestamp generation, and model saving.
'''
from typing import Dict, Optional, Tuple, Union, List
import os
import random
from glob import glob

import torch
from torch.utils.data import DataLoader

from yw_basics.dataloader import ImageClassificationDataset
from yw_basics.utils import current_datetime, import_object

from ..models.prior.base import AbstractVQLatentPriorModel

def get_image_files(filepaths : List[str]):
    '''
        process file paths to find image files
    '''
    img_files = []
    label_files = []
    for f_ in filepaths:
        if os.path.isfile(f_):
            if ImageClassificationDataset.is_image_file(f_):
                img_files.append(f_)
            else:
                label_files.append(f_)
        elif os.path.isdir(f_):
            files = glob(os.path.join(f_, "*.*"))
            img_f, lb_f = get_image_files(files)
            img_files.extend(img_f)
            label_files.extend(lb_f)
    return img_files, label_files

def get_datasets(data_cfg : Dict):
    '''
        Create torch dataset.
    '''
    train_imgs, train_lbs = get_image_files(data_cfg['train_files'])
    val_imgs, val_lbs = get_image_files(data_cfg['validation_files'])
    train_data = ImageClassificationDataset(
        label_files = train_lbs,
        image_files = train_imgs,
        transforms_configs = data_cfg.get('transforms', []),
        normalization_option = data_cfg.get('normalization', None),
        colorspace=data_cfg.get('colorspace', None)
    )
    val_data = ImageClassificationDataset(
        label_files = val_lbs,
        image_files = val_imgs,
        transforms_configs = data_cfg.get('transforms', []),
        normalization_option = data_cfg.get('normalization', None),
        colorspace=data_cfg.get('colorspace', None)
    )
    if not (
        train_data.normalization_option is None or
        isinstance(train_data.normalization_option, str) and
        train_data.normalization_option.lower() == 'none'
    ):
        raise ValueError(
            'Normalization option for training dataset must be none'
        )
    if train_data.normalization_option != val_data.normalization_option:
        raise ValueError(
            'Normalization options for training and validation datasets must be the same.'
        )
    print(f'Load {len(train_data)} images for training and {len(val_data)} images for validation')
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
        val_results : Dict,
        keyword : str = ''
    ) -> str:
    '''
        Save the model state and training results to a file.
    '''
    results_to_save = {
        'model': model.state_dict(),
        'config': model_cfg,
        'train_results': train_results,
        'validation_results': val_results
    }
    os.makedirs(save_path, exist_ok=True)
    fn = os.path.join(
        save_path, f'{keyword}_' + current_datetime() + '.pth'
    )
    torch.save(results_to_save, fn)
    return fn

def load_model_from_state_dict(
        state_dict : Union[Dict, str],
        model_type : Optional[str],
        config : Optional[Dict] = None
    ) -> Tuple[torch.nn.Module, Optional[Dict]]:
    '''
        load model from the a .pt file
    '''
    if isinstance(state_dict, str):
        state_dict = torch.load(state_dict, weights_only=False)

    if config is None:
        config = state_dict.get('config', None)

    model_cfg = config['model']
    if isinstance(model_type, str) and len(model_type) > 0:
        model_type = import_object(model_type)
    else:
        model_type = import_object(model_cfg.pop('type'))

    if issubclass(model_type, AbstractVQLatentPriorModel):
        encoder_cfg = config['encoder']
        encoder_wgts = torch.load(
            encoder_cfg['checkpoint'],
            weights_only = False
        )
        encoder, _ = load_model_from_state_dict(
            encoder_wgts, None
        )
        model = model_type(
            feature_extractor_model = encoder,
            **model_cfg
        )
    else:
        model = model_type(**model_cfg)

    model.load_state_dict(state_dict['model'], strict=True)

    return model, config

def try_parse_number(string : str):
    ''' try to parse a string as a number, int or float '''
    try:
        val = float(string)
    except ValueError:
        return string
    int_val = int(val)
    if val - int_val == 0:
        return int_val
    return val

def parse_kwargs(kwargs : List[str]) -> Dict:
    ''' argparse kwargs '''
    return {
        k : try_parse_number(v)
        for k, v in dict(kv.split('=') for kv in kwargs).items()
    }
