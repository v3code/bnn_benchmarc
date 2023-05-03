from typing import Optional

import pyro
import pyro.distributions as dist
import torch
from ml_collections import ConfigDict
from pyro.nn import PyroModule, PyroSample
from torch import nn

from bnn.utils.activation_factory import get_activation
from bnn.utils.types import ActivationsType, WeightDistributions
from bnn.utils.weight_dist_factory import get_weight_distribution


class BNNClassifier(PyroModule):
    def __init__(self, in_dim: int, out_dim: int, dist_config: ConfigDict, hidden_dim: int = 256,
                 activation: ActivationsType = 'relu', num_layers: int = 3, use_weight_scale: bool = True,
                 weight_distribution: WeightDistributions = 'normal', dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None):
        assert in_dim > 0 and out_dim > 0 and hidden_dim > 0 and num_layers > 0  # make sure the dimensions are valid

        super().__init__()
        self.activation = get_activation(activation)

        self.norm = nn.LayerNorm(in_dim)
        # Define the layer sizes and the PyroModule layer list
        layer_sizes = [in_dim] + num_layers * [hidden_dim] + [out_dim]
        layer_list = [PyroModule[nn.Linear](layer_sizes[idx - 1], layer_sizes[idx]) for idx in
                      range(1, len(layer_sizes))]
        self.layers = PyroModule[nn.ModuleList](layer_list)

        for layer_idx, layer in enumerate(self.layers):
            size_scale = layer_sizes[layer_idx] if use_weight_scale else None
            layer.weight = PyroSample(get_weight_distribution(weight_distribution, dist_config, size_scale, dtype=dtype,
                                                              device=device)
                                      .expand(layer.weight.shape)
                                      .to_event(layer.weight.dim()))
            layer.bias = PyroSample(get_weight_distribution(weight_distribution, dist_config, dtype=dtype,
                                                            device=device)
                                    .expand(layer.bias.shape)
                                    .to_event(layer.bias.dim()))

    def forward(self, x: torch.Tensor, obs: Optional[torch.Tensor] = None):
        x_hat = self.norm(x)
        for layer in self.layers:
            x_hat = self.activation(layer(x))
        with pyro.plate('data', x.shape[0]):
            pyro.sample('obs', dist.Categorical(logits=x_hat), obs=obs)
        return x_hat



