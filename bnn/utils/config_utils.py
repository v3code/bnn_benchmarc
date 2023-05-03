import importlib.util
import os.path


def create_conv_config(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        padding_mode: str = 'zeros',
        bias: bool = True
):
    return dict(in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                padding_mode=padding_mode,
                bias=bias)



def load_config(config_name):
    config_path = os.path.join(os.getcwd(), 'bnn', 'config', f'{config_name}.py')
    if not os.path.exists(config_path):
        raise ValueError(f'Config {config_name} does not exist')

    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    if not hasattr(config_module, 'get_config'):
        raise ValueError('Every config module should contain get_config function which returns config itself')
    return config_module.get_config()

