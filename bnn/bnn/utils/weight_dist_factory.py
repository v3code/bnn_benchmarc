import copy
from typing import Optional

import numpy as np
import torch
import pyro.distributions as dist
from ml_collections import ConfigDict

from bnn.utils.types import WeightDistributions


def get_weight_distribution(
        distribution: WeightDistributions,
        dist_config: ConfigDict,
        size_scale: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
):
    distribution = distribution.lower()
    dist_config = copy.deepcopy(dist_config.to_dict())
    if distribution == 'uniform':
        # For some reason PyroSample need this in order to work with cuda
        dist_config['low'] = torch.tensor(dist_config['low'], dtype=dtype, device=device)
        return dist.Uniform(**dist_config)
    elif distribution == 'normal':
        dist_config['scale'] = torch.tensor(dist_config['scale'], dtype=dtype, device=device)
        if size_scale:
            dist_config['scale'] = dist_config['scale'] * np.sqrt(2 / size_scale)
        return dist.Normal(**dist_config)
    elif distribution == 'soft-laplace':
        dist_config['scale'] = torch.tensor(dist_config['scale'], dtype=dtype, device=device)
        if size_scale:
            dist_config['scale'] = dist_config['scale'] * np.sqrt(2 / size_scale)
        return dist.SoftLaplace(**dist_config)
    elif distribution == 'asymmetric-laplace':
        dist_config['scale'] = torch.tensor(dist_config['scale'], dtype=dtype, device=device)
        if size_scale:
            dist_config['scale'] = dist_config['scale'] * np.sqrt(2 / size_scale)
        return dist.AsymmetricLaplace(**dist_config)
    elif distribution == 'sa-laplace':
        dist_config['scale'] = torch.tensor(dist_config['scale'], dtype=dtype, device=device)
        if size_scale:
            dist_config['scale'] = dist_config['scale'] * np.sqrt(2 / size_scale)
        return dist.SoftAsymmetricLaplace(**dist_config)
    else:
        raise ValueError(f"Distribution {distribution} is not supported")
