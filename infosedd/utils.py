import torch

import os
import logging
from omegaconf import OmegaConf, open_dict
import numpy as np

from torch.utils.data import Dataset

class SynthetitcDataset(Dataset):

    def __init__(self, data, device=None):
        self.data = data
        self.device = device
    
    def set_device(self, device):
        self.device = device
        self.data = self.data.to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def _array_to_tensor(x, dtype=torch.int64):
    x = np.array(x)
    if not isinstance(x, np.ndarray):
        raise TypeError("Input must be a numpy array or convertible to one.")
    return torch.tensor(x, dtype=dtype)

def array_to_dataset(x: np.array, y: np.array, device=None):
    x = _array_to_tensor(x)
    y = _array_to_tensor(y)
    data = torch.cat((x, y), dim=1)
    if device is not None:
        data = data.to(device)
    if data.shape[-1] == 1:
        data = data.squeeze(-1)
    dataset = SynthetitcDataset(data)
    return dataset

def statistics_batch(x):
    hist = np.zeros((torch.max(x)+1, torch.max(x)+1))
    for s in x:
        hist[s[0], s[1]] += 1
    hist = hist / hist.sum()
    raise UserWarning(f"Histogram of samples: {hist}, number of unique samples: {len(torch.unique(x))}, n_samples: {len(x)}")

def load_hydra_config_from_run(load_dir):
    cfg_path = os.path.join(load_dir, ".hydra/config.yaml")
    cfg = OmegaConf.load(cfg_path)
    return cfg


def makedirs(dirname):
    os.makedirs(dirname, exist_ok=True)


def get_logger(logpath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    if (logger.hasHandlers()):
        logger.handlers.clear()

    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        info_file_handler.setFormatter(formatter)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


def restore_checkpoint(ckpt_dir, state, device):
    if not os.path.exists(ckpt_dir):
        makedirs(os.path.dirname(ckpt_dir))
        logging.warning(f"No checkpoint found at {ckpt_dir}. Returned the same state as input")
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        state['optimizer'].load_state_dict(loaded_state['optimizer'])
        state['model'].module.load_state_dict(loaded_state['model'], strict=False)
        state['ema'].load_state_dict(loaded_state['ema'])
        state['step'] = loaded_state['step']
        return state
    
def cartesian_power(tensor, n):
    """
    Compute the Cartesian power of a tensor.
    
    Args:
        tensor (torch.Tensor): The input tensor.
        n (int): The number of times to take the Cartesian product of the tensor with itself.
    
    Returns:
        torch.Tensor: The Cartesian power of the input tensor.
    """
    grids = [tensor] * n
    meshgrids = torch.meshgrid(*grids, indexing='ij')
    cartesian_product = torch.stack(meshgrids, dim=-1).reshape(-1, n)
    return cartesian_product


def save_checkpoint(ckpt_dir, state):
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': state['model'].module.state_dict(),
        'ema': state['ema'].state_dict(),
        'step': state['step']
    }
    torch.save(saved_state, ckpt_dir)