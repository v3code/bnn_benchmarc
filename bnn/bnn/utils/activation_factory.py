from torch import nn

from bnn.utils.types import ActivationsType


def get_activation(activation: ActivationsType):
    activation = activation.lower()
    if activation == 'tanh':
        return nn.Tanh()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'elu':
        return nn.ELU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation == 'celu':
        return nn.CELU()
    elif activation == 'selu':
        return nn.SELU()
    elif activation == 'hardswish':
        return nn.Hardswish()
    else:
        raise ValueError(f"Activation '{activation}' is not supported")
