import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader, RandomSampler
import os

# ------------------ LOCAL IMPORTS ---------------------------------
from Maestro.data.HuggingFaceDataset import HuggingFaceDataset
from Maestro.data.TorchVisionDataset import TorchVisionDataset
# ------------------ LOCAL IMPORTS ---------------------------------


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
    elif name == "Malimg":
        return _read_malimg_dataset()



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
    print(train_data)

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


def _read_malimg_dataset(train_size_ratio=0.7):
    torch.manual_seed(0)
    data_dir = '../../data/Malimg/'
    print("data path: ", data_dir)
    if not os.path.exists(data_dir): os.makedirs(data_dir)
    data_transform = transforms.Compose([
        transforms.CenterCrop((216, 64)),
        transforms.ToTensor(),])
    malimg_datasets = datasets.ImageFolder(root=data_dir, transform=data_transform)
    dataset_size = len(malimg_datasets)
    train_size = int(dataset_size * train_size_ratio)
    test_size = dataset_size - train_size
    train_data, test_data = torch.utils.data.random_split(malimg_datasets, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    # print(train_data, test_data)
    # print(malimg_datasets)
    return {"train": train_data, "test": test_data}
