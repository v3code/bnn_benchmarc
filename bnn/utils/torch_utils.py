import os.path
from typing import Optional

import numpy.random
import pyro
import torch
from pyro.infer.autoguide import AutoGuide
from pyro.nn import PyroModule
from pyro.optim import PyroOptim


def get_auto_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_seed(seed: int):
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)
    numpy.random.seed(seed)


def init_svi_method(model: PyroModule,
                    optim: PyroOptim,
                    guide: AutoGuide,
                    checkpoint_path: Optional[str] = None):
    epoch = 0
    step = 0

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        epoch = checkpoint['epoch']
        step = checkpoint['step']
        model.load_state_dict(checkpoint['model_state'])
        optim.set_state(checkpoint['optim_state'])
        guide.load_state_dict(checkpoint['guide_state'])

    return epoch, step


def save_svi_checkpoint(checkpoint_path: str, epoch: int, step: int, model: PyroModule, optim: PyroOptim, guide: AutoGuide):
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    torch.save(dict(
        epoch=epoch,
        step=step,
        model_state=model.state_dict(),
        optim_state=optim.get_state(),
        guide_state=guide.state_dict(),
    ), os.path.join(checkpoint_path, f'model_{epoch}_{step}.ckpt'))
