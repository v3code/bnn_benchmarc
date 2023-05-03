from ml_collections import ConfigDict
from pyro.infer.autoguide import AutoNormal, AutoDelta, AutoContinuous, AutoMultivariateNormal, AutoDiagonalNormal, \
    AutoLowRankMultivariateNormal, AutoNormalizingFlow, AutoGaussian
from pyro.nn import PyroModule


def get_guide(config: ConfigDict, model: PyroModule):
    guide_name = config.guide_name.lower()
    if guide_name == 'normal':
        return AutoNormal(model, **config.guide)
    elif guide_name == 'delta':
        return AutoDelta(model, **config.guide)
    elif guide_name == 'continuous':
        return AutoContinuous(model, **config.guide)
    elif guide_name == 'normal-multivar':
        return AutoMultivariateNormal(model, **config.guide)
    elif guide_name == 'normal-diagonal':
        return AutoDiagonalNormal(model, **config.guide)
    elif guide_name == 'normal-multivar-lowrank':
        return AutoLowRankMultivariateNormal(model, **config.guide)
    elif guide_name == 'gaussian':
        return AutoGaussian(model, **config.guide)
    else:
        raise ValueError(f'Guide "{guide_name}" is not supported')