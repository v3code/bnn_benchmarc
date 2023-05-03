import os.path

import wandb
from ml_collections import ConfigDict

from bnn.config.base_config import base_cfg
from bnn.utils.config_utils import create_conv_config


def get_config():
    cfg = ConfigDict(base_cfg)

    cfg.name = 'bcnn_mnist_svi_norm'
    cfg.dataset = 'mnist'
    cfg.model_name = 'bcnn'

    cnn_configs = (
        create_conv_config(
            in_channels=1,
            out_channels=16,
            kernel_size=5,
        ),
        create_conv_config(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            bias=False,
            padding=1
        ),
    )

    cfg.model.conv_configs = cnn_configs
    cfg.model.num_classes = 10
    cfg.model.head_in_dim = 16 * 8 * 8

    cfg.model.dist_config = dict(
        loc=0.,
        scale=1.
    )

    cfg.model.weight_distribution = 'normal'
    cfg.model.use_residuals = (False, False, True, True)
    cfg.model.head_hidden_dim = 128
    cfg.model.head_activation = 'hardswish'


    cfg.loggers.wandb = True
    cfg.wandb.name = 'BCNN MNIST SVI Normal'
    cfg.wandb.dir = os.path.join('logs', cfg.name)

    cfg.checkpoint_root = os.path.join('checkpoints', cfg.name)

    cfg.lock()
    return cfg
