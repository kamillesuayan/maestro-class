import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import textattack
from Maestro.data.HuggingFaceDataset import HuggingFaceDataset
from Maestro.data.TorchVisionDataset import TorchVisionDataset
import numpy as np
from torch.utils.data import DataLoader, RandomSampler


def get_data(name: str):
    vocab = None
    if name == "MNIST":
        return _read_mnist_dataset()
        # test_loader = torch.utils.data.DataLoader(
        #     datasets.MNIST(
        #         "../tmp",
        #         train=False,
        #         download=True,
        #         transform=transforms.Compose([transforms.ToTensor(),]),
        #     ),
        #     batch_size=1,
        #     shuffle=True,
        # )
        # train_loader = torch.utils.data.DataLoader(
        #     datasets.MNIST(
        #         "../tmp",
        #         train=True,
        #         download=True,
        #         transform=transforms.Compose([transforms.ToTensor(),]),
        #     ),
        #     batch_size=1,
        #     shuffle=True,
        # )
    elif name == "SST":
        return _read_sst_dataset()


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

    return {"train": train_data, "dev": test_data}


def _read_sst_dataset():
    train_data = HuggingFaceDataset(
        name="glue", subset="sst2", split="train", label_map=None, shuffle=True
    )
    validation_data = HuggingFaceDataset(
        name="glue", subset="sst2", split="test", label_map=None, shuffle=True
    )
    return train_data, validation_data

