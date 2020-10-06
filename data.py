import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

WRITE_ACESS = 1
READ_ACESS = 2
READ_WRITE_ACESS = 3


class DataModifier:
    """
    """

    def __init__(self, data: DataLoader, access: int) -> None:
        self.data = data
        self.access = access

    def read(self, idx):
        if self.access != 3 or self.access != 1:
            raise ValueError("Do not have access to read data")
        return self.data[idx]

    def write(self, type, idxes, values):
        if self.access != 3 or self.access != 2:
            raise ValueError("Do not have access to write data")
        if type == "front":
            self.data[idxes] = torch.cat(values, self.data[idxes])
        elif type == "back":
            self.data[idxes] = torch.cat(self.data[idxes], values)


def get_data(name: str):
    if name == "MNIST":
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "../data",
                train=False,
                download=True,
                transform=transforms.Compose([transforms.ToTensor(),]),
            ),
            batch_size=1,
            shuffle=True,
        )
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "../data",
                train=True,
                download=True,
                transform=transforms.Compose([transforms.ToTensor(),]),
            ),
            batch_size=1,
            shuffle=True,
        )
        dev_loader = None
    return train_loader, dev_loader, test_loader
