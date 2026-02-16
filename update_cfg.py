''' Update config saved in a .pt file '''

from typing import Any
import json
import argparse
import sys

import torch
    
def update_cfg(
        model_file : str,
        keychain : str,
        value : Any
    ):
    '''update config in a .pt file'''
    dat = torch.load(model_file, weights_only = False)
    cfg = dat.get('config', None)

    if not isinstance(cfg, dict):
        raise RuntimeError('Configuration is nonexistent')

    chain = keychain.split('.')
    for p_ in chain[:-1]:
        cfg = cfg[p_]

    cfg[chain[-1]] = value
    torch.save(dat, model_file)

def print_cfg(model_file : str):
    ''' display config '''
    dat = torch.load(model_file, weights_only = False)
    cfg = dat.get('config', None)

    if not isinstance(cfg, dict):
        raise RuntimeError('Configuration is nonexistent')

    print(json.dumps(cfg, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser(description = "Change config in a .pt file.")
    parser.add_argument(
        'filename', help = "Path to the .pt file."
    )
    parser.add_argument(
        '-k', '--keychain', help = 'Chain of key to the value to be changed'
    )
    parser.add_argument(
        '-v', '--value', help = 'New vlaue to be updated'
    )

    if len(sys.argv) == 1:
        sys.argv.append("-h")

    args = parser.parse_args()

    if args.keychain is None or args.value is None:
        print_cfg(args.filename)
        print('Add keychain and value to update the above')
        sys.exit(0)

    update_cfg(args.filename, args.keychain, args.value)
