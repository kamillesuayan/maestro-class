import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import textattack
from Maestro.data.HuggingFaceDataset import HuggingFaceDataset
from Maestro.data.TorchVisionDataset import TorchVisionDataset
import numpy as np
from torch.utils.data import DataLoader, RandomSampler


def get_dataset(name: str):
    vocab = None
    if name == "MNIST":
        return _read_mnist_dataset()
    elif name == "SST2":
        return _read_sst_dataset()
    elif name == "IMDB":
        return _read_imdb_dataset()
    elif name == "MNLI":
        return _read_mnli_dataset()


def _read_mnist_dataset():
    train_data = TorchVisionDataset(
        name="mnist",
        split="train",
        transforms=transforms.Compose([transforms.ToTensor(),]),
    )
    test_data = TorchVisionDataset(
        name="mnist",
        split="test",
        transforms=transforms.Compose([transforms.ToTensor(),]),
    )

    return {"train": train_data, "test": test_data}


def _read_sst_dataset():
    train_data = HuggingFaceDataset(
        name="glue", subset="sst2", split="train", label_map=None, shuffle=True
    )
    validation_data = HuggingFaceDataset(
        name="glue", subset="sst2", split="validation", label_map=None, shuffle=True
    )
    test_data = HuggingFaceDataset(
        name="glue", subset="sst2", split="test", label_map=None, shuffle=True
    )
    return train_data, validation_data, test_data


def _read_imdb_dataset():
    train_data = HuggingFaceDataset(
        name="imdb", subset=None, split="train", label_map=None, shuffle=True
    )
    test_data = HuggingFaceDataset(
        name="imdb", subset=None, split="test", label_map=None, shuffle=True
    )
    return train_data, test_data


def _read_mnli_dataset():
    train_data = HuggingFaceDataset(
        name="glue", subset="mnli", split="train", label_map=None, shuffle=True
    )
    validation_data = HuggingFaceDataset(
        name="glue",
        subset="mnli",
        split="validation_method",
        label_map=None,
        shuffle=True,
    )
    test_data = HuggingFaceDataset(
        name="glue", subset="mnli", split="test", label_map=None, shuffle=True
    )
    return train_data, validation_data, test_data

