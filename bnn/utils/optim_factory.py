from ml_collections import ConfigDict
from pyro.nn import PyroModule
from pyro.optim import ClippedAdam, PyroOptim
from torch.optim import Adam, AdamW


def get_optim(config: ConfigDict):
    optim_name = config.optim_name.lower()
    if optim_name == 'adam':
        return PyroOptim(Adam, **config.optim.to_dict())
    elif optim_name == 'adamw':
        return PyroOptim(AdamW, **config.optim.to_dict())
    elif optim_name == 'clipped-adam':
        return ClippedAdam(**config.optim.to_dict)
    else:
        raise ValueError(f'Optimizer by name "{optim_name}" is not supported')
