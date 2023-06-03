import torch
from ml_collections import ConfigDict
from torchvision.transforms import transforms

from bnn.utils.torch_utils import get_auto_device

base_cfg = ConfigDict()
base_cfg.name = 'base_cfg'

base_cfg.seed = 42

base_cfg.project = 'BCNN Benchmark'

base_cfg.epochs = 10

base_cfg.device = get_auto_device()
base_cfg.use_half_precision = False

base_cfg.batch_size = 32
base_cfg.dataset = ''
base_cfg.dataset_root = 'data/'
base_cfg.dataset_train_transforms = transforms.ToTensor()
base_cfg.dataset_val_transforms = transforms.ToTensor()
base_cfg.num_workers = 1
base_cfg.pbar_step = 5

base_cfg.log_step = 25

base_cfg.optim_name = 'adamw'
base_cfg.optim = ConfigDict()
base_cfg.optim.optim_args = dict(lr=1e-1)

base_cfg.infer_type = 'svi'
base_cfg.loss_config = ConfigDict()
base_cfg.loss_jit = False

base_cfg.predict_num_samples = 10

base_cfg.guide_name = 'normal'
base_cfg.guide = ConfigDict()

base_cfg.model_name = ''
base_cfg.model = ConfigDict()
base_cfg.model.dist_config = ConfigDict()

base_cfg.loggers = ConfigDict()
base_cfg.loggers.wandb = True

base_cfg.wandb = ConfigDict()
base_cfg.wandb.project = base_cfg.project

base_cfg.use_exp_lr = False
base_cfg.lr_config = ConfigDict()


base_cfg.nuts_config = ConfigDict()
base_cfg.mcmc_config = ConfigDict()

base_cfg.max_batches = float('inf')
