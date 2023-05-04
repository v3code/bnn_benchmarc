import os.path

import wandb
from ml_collections import ConfigDict

from bnn.config.base_config import base_cfg
from bnn.utils.config_utils import create_conv_config
from torchvision.transforms import transforms
import torch


def get_config():
    cfg = ConfigDict(base_cfg)

    cfg.name = 'bcnn_omniglot_svi_uniform'
    cfg.dataset = 'omniglot'
    cfg.model_name = 'bcnn'

    cfg.num_workers = 1
    
    os.environ["CUDA_AVAILABLE_DEVICES"] = '1'
    
    cnn_configs = (
        create_conv_config(
            in_channels=1,
            out_channels=128,
            kernel_size=5,
        ),
        create_conv_config(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            padding=1
        ),
        create_conv_config(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            padding=1
        ),
        create_conv_config(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            padding=1
        ),
    )

    cfg.epochs = 9

    cfg.num_classes = 1623

    cfg.model.conv_configs = cnn_configs
    cfg.model.num_classes = cfg.num_classes
    cfg.model.head_in_dim = 128 * 9 * 9
    cfg.batch_size = 32
    
    cfg.dataset_train_transforms = transforms.Compose([
        transforms.RandomRotation(45),
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    
    cfg.dataset_val_transforms = transforms.Compose([
        transforms.RandomRotation(45),
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    cfg.optim.optim_args = dict(lr=1e-2)

    cfg.loss_jit = True

    cfg.model.weight_distribution = 'uniform'
    cfg.model.use_residuals = (False, False, True, True)
    cfg.model.head_hidden_dim = 2042
    cfg.model.head_activation = 'selu'
    cfg.guide_name = 'normal-diagonal'
    cfg.model.dist_config = dict(
        low=-1.,
        high=1.,
    )
    
    cfg.use_exp_lr = True
    cfg.lr_config = dict(gamma=0.1)

    cfg.predict_num_samples = 100

    cfg.loggers.wandb = True
    cfg.wandb.name = 'BCNN Omniglot SVI Uniform'
    cfg.wandb.dir = os.path.join('logs', cfg.name)

    cfg.checkpoint_root = os.path.join('checkpoints', cfg.name)

    cfg.lock()
    return cfg
