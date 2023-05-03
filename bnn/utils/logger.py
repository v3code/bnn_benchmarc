from typing import Dict, Optional

import wandb
from ml_collections import ConfigDict


def create_logger(config: ConfigDict):
    context = {}

    def on_start():
        if config.loggers.wandb:
            context['wandb_run'] = wandb.init(**config.wandb)

    def log(log_dict: Dict, step: int, validation: Optional[bool] = False):
        if step % config.log_step == 0 and not validation:
            return

        if 'wandb_run' in context:
            context['wandb_run'].log(log_dict, step=step)

    def on_end():
        if 'wandb_run' in context:
            context['wandb_run'].finish()

    return on_start, log, on_end
