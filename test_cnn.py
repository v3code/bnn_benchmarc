import torch
from torch import nn

from bnn.utils import create_conv_config

cnn_configs = (
        create_conv_config(
            in_channels=3,
            out_channels=16,
            kernel_size=5,
        ),
        create_conv_config(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            bias=False,
            padding=1
        ),
        create_conv_config(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            bias=False,
            padding=1,
        ),
        create_conv_config(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            bias=False,
            padding=1,
        ),
    )

test_image = torch.rand((2, 3, 28, 28))
cnn_layers = []
for c in cnn_configs:
    cnn_layers.append(nn.Conv2d(**c))

cnn = nn.Sequential(*cnn_layers)
max_pool = nn.MaxPool2d(3)

out = max_pool(cnn(test_image))

print(out.shape)