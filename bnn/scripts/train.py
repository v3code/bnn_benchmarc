import argparse

from bnn.methods.factory import get_method
from bnn.utils.config_utils import load_config
from bnn.utils.logger import create_logger

parser = argparse.ArgumentParser("BNN Training Script")

parser.add_argument('--config')
parser.add_argument('--checkpoint', default=None)


def main():
    args = parser.parse_args()
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
