from ml_collections import ConfigDict

from bnn.modules.bcnn import BCNNClassifier
from bnn.modules.bnn import BNNClassifier


def get_classifier_model(config: ConfigDict()):
    name = config.model_name.lower()
    if name == 'bnn':
        return BNNClassifier(**config.model, device=config.device)
    if name == 'bcnn':
        return BCNNClassifier(**config.model, device=config.device)
    else:
        raise ValueError(f"Model {name} does not exist")