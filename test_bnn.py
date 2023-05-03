import torch

from bnn.config.bcnn_mnist_svi_norm import get_config
from bnn.utils import get_classifier_model

test_image = torch.rand((2, 3, 28, 28))

model_config = get_config()

model = get_classifier_model(model_config)

print(model(test_image))
