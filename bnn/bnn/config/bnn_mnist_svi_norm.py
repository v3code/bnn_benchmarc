import os.path

import wandb
from ml_collections import ConfigDict

from bnn.config.base_config import base_cfg
from bnn.utils.config_utils import create_conv_config


def get_config():
    cfg = ConfigDict(base_cfg)

    cfg.name = 'bnn_mnist_svi_norm'
    cfg.dataset = 'mnist'
    cfg.model_name = 'bnn'

    cfg.num_workers = 3

    cfg.num_classes = 10

    cfg.epochs = 10

    cfg.batch_size = 128

    cfg.model.dist_config = dict(
        loc=0.,
        scale=1.
    )

    cfg.optim.optim_args = dict(lr=1e-2)

    # cfg.loss_config = dict(
    #     num_particles15=
    # )
    
    cfg.guide_name = 'normal-diagonal'
    

    cfg.model.out_dim = 10
    cfg.model.in_dim = 28 * 28
    cfg.model.weight_distribution = 'normal'
    cfg.model.hidden_dim = 1024
    cfg.model.num_layers = 3
    cfg.model.activation = 'selu'

    cfg.predict_num_samples = 500

    cfg.loggers.wandb = True
    cfg.wandb.name = 'BNN MNIST SVI Normal'
    cfg.wandb.dir = os.path.join('logs', cfg.name)

    cfg.checkpoint_root = os.path.join('checkpoints', cfg.name)

    cfg.lock()
    return cfg
