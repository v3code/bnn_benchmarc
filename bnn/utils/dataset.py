import os

from torchvision.datasets import MNIST, Omniglot, LFWPeople
from torch.utils.data import Dataset, DataLoader
from ml_collections import ConfigDict


def create_dataloaders(train_dataset: Dataset, val_dataset: Dataset, config: ConfigDict):
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                  num_workers=config.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=config.num_workers)
    return train_dataloader, val_dataloader


def get_dataset(config: ConfigDict):
    dataset = config.dataset.lower()
    train_root = os.path.join(config.dataset_root, 'train')
    val_root = os.path.join(config.dataset_root, 'val')
    if dataset == 'mnist':
        train_dataset = MNIST(train=True, transform=config.dataset_train_transforms, download=True,
                              root=train_root)
        val_dataset = MNIST(train=False, transform=config.dataset_val_transforms, download=True,
                            root=val_root)

        return train_dataset, val_dataset
    elif dataset == 'omniglot':
        train_dataset = Omniglot(transform=config.dataset_train_transforms, download=True,
                                 root=train_root)
        val_dataset = Omniglot(background=True, transform=config.dataset_val_transforms, download=True,
                               root=val_root)

        return train_dataset, val_dataset
    else:
        raise ValueError(f"Dataset '{dataset}' is not implemented")
