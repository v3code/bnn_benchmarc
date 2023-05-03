from typing import Union

from ml_collections import ConfigDict
from pyro.infer import Trace_ELBO, SVI, MCMC, NUTS
from pyro.nn import PyroModule
from pyro.optim import PyroOptim
from torch.optim import Optimizer


def get_kernel(config: ConfigDict, model: PyroModule):
    kernel = config.mcmc_kernel_name.lower()
    if kernel == 'nuts':
        return NUTS(model, **config.mcmc_kernel)
    # elif kernel == ''


def get_infer(config: ConfigDict, model: PyroModule, guide: PyroModule, optim: Union[PyroOptim, Optimizer]):
    infer_name = config.infer_name.lower()
    if infer_name == 'svi':
        loss = Trace_ELBO(**config.loss_config)
        return SVI(model, guide, optim, loss)
    if infer_name == 'mcmc':
        kernel = get_kernel(config, model)
        return MCMC(kernel, **config.infer)

