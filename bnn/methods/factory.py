from ml_collections import ConfigDict

from bnn.methods.classification import classification_svi_method


def get_method(config: ConfigDict):
    infer_type = config.infer_type.lower()
    if infer_type == 'svi':
        return classification_svi_method
    else:
        ValueError(f'Inference type "{infer_type}" is not available')