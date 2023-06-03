import argparse

import numpy.random
import pyro
import torch

from bnn.methods.factory import get_method
from bnn.utils.config_utils import load_config
from bnn.utils.logger import create_logger

parser = argparse.ArgumentParser("BNN Training Script")
parser.add_argument('--seed', type=int, default=42)


benchmark_configs = ('bcnn_mnist_svi_laplace', 'bcnn_mnist_svi_norm', 'bcnn_mnist_svi_uniform', 'bcnn_mnist_svi_laplace_small', 'bcnn_mnist_svi_laplace_large')

def main():

    args = parser.parse_args()

    pyro.set_rng_seed(args.seed)
    torch.manual_seed(args.seed)
    numpy.random.seed(args.seed)
    for config_name in benchmark_configs:
        print(f'benchmarking {config_name}')

        config = load_config(args.config)

        on_start_logger, log, on_end_logger = create_logger(config)
        method = get_method(config)

        on_start_logger()
        try:
            method(config, log, args.checkpoint)
        finally:
            on_end_logger()


if __name__ == "__main__":
    main()
