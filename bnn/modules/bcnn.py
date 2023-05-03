from typing import Optional, Sequence

import pyro
import pyro.distributions as dist
import torch
from ml_collections import ConfigDict
from pyro.distributions.transforms import BatchNorm
from pyro.nn import PyroModule, PyroSample
from torch import nn

from bnn.utils.activation_factory import get_activation
from bnn.utils.types import ActivationsType, WeightDistributions
from bnn.utils.weight_dist_factory import get_weight_distribution


class BCNNClassifier(PyroModule):
    def __init__(self,
                 conv_configs: Sequence,
                 dist_config: ConfigDict,
                 num_classes: int,
                 head_in_dim: int,
                 head_hidden_dim: int = 245,
                 activation: ActivationsType = 'relu',
                 head_activation: ActivationsType = 'gelu',
                 weight_distribution: WeightDistributions = 'normal',
                 use_residuals: Sequence[bool] = (),
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None):

        super().__init__()
        self.activation = get_activation(activation)
        self.head_activation = get_activation(head_activation)

        self.use_residuals = use_residuals

        conv_layers = []
        norm_layers = []
        for conv_config in conv_configs:
            conv_layers.append(PyroModule[nn.Conv2d](**conv_config))
            norm_layers.append(nn.BatchNorm2d(conv_config['in_channels']))

        self.conv_layers = PyroModule[nn.ModuleList](conv_layers)
        self.norm_layers = nn.ModuleList(norm_layers)

        self.max_pool = nn.MaxPool2d(3)

        self.head_norm = nn.LayerNorm(head_in_dim)
        self.head_fc1 = PyroModule[nn.Linear](head_in_dim, head_hidden_dim)
        self.head_fc2 = PyroModule[nn.Linear](head_hidden_dim, num_classes)

        for layer_idx, layer in enumerate(self.conv_layers):
            layer.weight = PyroSample(get_weight_distribution(weight_distribution, dist_config, dtype=dtype,
                                                              device=device)
                                      .expand(layer.weight.shape)
                                      .to_event(layer.weight.dim()))
            if layer.bias is not None:
                layer.bias = PyroSample(get_weight_distribution(weight_distribution, dist_config, dtype=dtype,
                                                                device=device)
                                        .expand(layer.bias.shape)
                                        .to_event(layer.bias.dim()))

        self.head_fc1.weight = PyroSample(get_weight_distribution(weight_distribution, dist_config, dtype=dtype,
                                                                  device=device)
                                          .expand(self.head_fc1.weight.shape)
                                          .to_event(self.head_fc1.weight.dim()))

        self.head_fc1.bias = PyroSample(get_weight_distribution(weight_distribution, dist_config, dtype=dtype,
                                                                device=device)
                                        .expand(self.head_fc1.bias.shape)
                                        .to_event(self.head_fc1.bias.dim()))

        self.head_fc2.weight = PyroSample(get_weight_distribution(weight_distribution,
                                                                  dist_config,
                                                                  dtype=dtype,
                                                                  device=device)
                                          .expand(self.head_fc2.weight.shape)
                                          .to_event(self.head_fc2.weight.dim()))

        self.head_fc2.bias = PyroSample(get_weight_distribution(weight_distribution, dist_config, dtype=dtype,
                                                                device=device)
                                        .expand(self.head_fc2.bias.shape)
                                        .to_event(self.head_fc2.bias.dim()))

    def forward(self, x: torch.Tensor, obs: Optional[torch.Tensor] = None):
        x_hat = x
        x_prev = x
        for conv_idx, conv in enumerate(self.conv_layers):
            x_hat = self.norm_layers[conv_idx](x_hat)
            x_hat = conv(x_hat)
            x_hat = self.activation(x_hat)
            if len(self.use_residuals) > conv_idx and self.use_residuals[conv_idx]:
                x_hat = x_hat + x_prev
            x_prev = x_hat

        x_hat = self.max_pool(x_hat)

        x_hat = torch.flatten(x_hat, 1)

        x_hat = self.head_norm(x_hat)
        x_hat = self.head_fc1(x_hat)
        x_hat = self.head_activation(x_hat)
        x_hat = self.head_fc2(x_hat)

        with pyro.plate('data', x.shape[0]):
            pyro.sample('obs', dist.Categorical(logits=x_hat), obs=obs)
        return x_hat
