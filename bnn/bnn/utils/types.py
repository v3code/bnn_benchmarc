from typing import Literal

ActivationsType = Literal['tanh', 'relu', 'gelu', 'elu', 'celu', 'selu', 'hardswish']
WeightDistributions = Literal['uniform', 'normal', 'soft-laplace', 'asymmetric-laplace', 'sa-laplace']