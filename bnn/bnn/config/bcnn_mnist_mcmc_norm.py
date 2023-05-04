import os.path
import torch

import wandb
from ml_collections import ConfigDict

from bnn.config.base_config import base_cfg
from bnn.utils.config_utils import create_conv_config


def get_config():
    cfg = ConfigDict(base_cfg)

    cfg.name = 'bcnn_mnist_mcmc_norm'
    cfg.dataset = 'mnist'
    cfg.model_name = 'bcnn'

    cfg.num_workers = 1

    cnn_configs = (
        create_conv_config(
            in_channels=1,
            out_channels=20,
            kernel_size=5,
        ),
        create_conv_config(
            in_channels=20,
            out_channels=20,
            kernel_size=3,
            padding=1
        ),
        create_conv_config(
            in_channels=20,
            out_channels=20,
            kernel_size=3,
            padding=1
        ),
    )

    cfg.epochs = 26

    cfg.num_classes = 10

    cfg.model.conv_configs = cnn_configs
    cfg.model.num_classes = 10
    cfg.model.head_in_dim = 20 * 8 * 8
    cfg.batch_size = 256

    cfg.model.dist_config = dict(
        loc=0.,
        scale=1.
    )

    cfg.model.weight_distribution = 'normal'
    cfg.model.use_residuals = (False, False, True)
    cfg.model.head_hidden_dim = 1024
    cfg.model.head_activation = 'selu'
    
    
    cfg.use_exp_lr = True
    cfg.lr_config = dict(gamma=0.1)

    cfg.predict_num_samples = 500

    cfg.loggers.wandb = True
    cfg.wandb.name = 'BCNN MNIST MCMC Normal'
    cfg.wandb.dir = os.path.join('logs', cfg.name)

    cfg.checkpoint_root = os.path.join('checkpoints', cfg.name)
    
    cfg.infer_type = 'mcmc'
    cfg.mcmc_config = dict(
        num_samples=50,
        num_chains=1,
        warmup_steps=8,
        
    )

    cfg.lock()
    return cfg
