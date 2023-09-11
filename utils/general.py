import random
import numpy as np
import torch
import importlib
import os
import json
import pathlib
import inspect
from pypesq import pesq
from pathlib import Path
from torchmetrics import SignalNoiseRatio
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    # cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    # np.random.seed(0) # change by shoham
    torch.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)


def get_instance(module_name, instance_name):
    module = importlib.import_module(module_name)
    obj = getattr(module, instance_name)
    return obj


def save_config_to_file(config, current_folder):
    with open(os.path.join(current_folder, 'config.json'), 'w') as config_file:
        d = dict(vars(config))
        d_new = check_dict(d)
        d_new.pop('scheduler_factory')
        json.dump(d_new, config_file)


def save_class_to_file(config, current_folder):
    with open(os.path.join(current_folder, 'config.json'), 'w') as config_file:
        d = dict(vars(config))
        # d.pop('scheduler_factory')
        json.dump(d, config_file)

def check_dict(d):
    d_new = {}
    for key, value in d.items():
        if isinstance(value, dict):
            value = check_dict(value)
        if isinstance(value, list):
            for i, val in enumerate(value):
                if isinstance(val, dict):
                    value[i] = check_dict(val)
        elif isinstance(value, (pathlib.PosixPath, torch.device)):
            value = str(value)
        elif isinstance(value, torch.Tensor):
            continue
        elif inspect.ismethod(value):
            value = value.__qualname__
        d_new[key] = value
    return d_new



def PESQ(src_array, adv_array, sr=16000, device='cpu',eps=0):
    src_array = (src_array + eps).to(device)
    adv_array = adv_array.detach().to(device)
    pesq_calc = 0.0
    vector_numbers = 3# src_array.shape[0]
    for i in range(vector_numbers):
        pesq_curr = pesq(src_array[i].detach(), adv_array[i].detach(), sr)
        pesq_calc += pesq_curr
    return pesq_calc / vector_numbers


def calculate_snr_github_direct_pkg(pred, label): # pred = adversarial, label = source data
    snr_func =  SignalNoiseRatio().to(device=device)
    snr = snr_func(pred, label)
    return snr


def load_from_npy(root_path, audio_type, file_name):
    file_path = os.path.join(root_path, audio_type, Path(file_name).stem)
    return np.load(f'{file_path}.npy')


def get_pert(pert_type, size):
    if pert_type == 'zeros':
        adv_pert = torch.zeros((1, size))
    elif pert_type == 'ones':
        adv_pert = torch.ones((1, size))
    elif pert_type == 'random':
        adv_pert = torch.FloatTensor(1, size).uniform_(-0.0001, 0.0001)
    elif pert_type == 'prev':
        adv_pert = torch.from_numpy(load_from_npy( os.path.join('.', 'data', 'uap_perturbation', 'ecapa'), 'snr', '100ep_100spk'))
    adv_pert.requires_grad_(True)
    return adv_pert
