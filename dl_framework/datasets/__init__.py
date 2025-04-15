from .base_dataset import BaseDataset
from .registry import DatasetRegistry
from .cifar10 import CIFAR10Dataset
from .voc_dataset import VOC2012Dataset
__all__ = [
    'BaseDataset',
    'DatasetRegistry',
    'CIFAR10Dataset',
    'VOC2012Dataset',
]
