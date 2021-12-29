import torch
from torchvision import datasets, transforms
from Maestro.data.HuggingFaceDataset import HuggingFaceDataset
from Maestro.data.TorchVisionDataset import TorchVisionDataset
import numpy as np
from torch.utils.data import DataLoader, RandomSampler
import os

def get_dataset(dataset_configs):
    vocab = None
    name = dataset_configs["name"]
    if name == "MNIST":
        return _read_mnist_dataset(dataset_configs,name)
    elif name == "SST2":
        return _read_sst_dataset()
    elif name == "IMDB":
        return _read_imdb_dataset()
    elif name == "MNLI":
        return _read_mnli_dataset()
    elif name == "Malimg":
        return _read_malimg_dataset()

def _split_by_labels(num_classes,train_data,server_number_sampled,train_server_path):
    subset_indices = []
    for i in range(num_classes):
        indices_xi = (train_data.targets==i).nonzero(as_tuple=True)[0]
        sampled_indices = np.random.choice(indices_xi,server_number_sampled,replace=False)
        subset_indices.extend(sampled_indices)
    train_server_subset = torch.utils.data.Subset(train_data, subset_indices)
    torch.save(train_server_subset,train_server_path)
    return train_server_subset

def _read_mnist_dataset(dataset_configs, dataset_name):
    path = dataset_configs["dataset_path"]

    # 1.1 Training Data for Server
    train_server_path = os.path.join(path, "train_server_split.pt")
    if os.path.exists(train_server_path):
        train_server_subset = torch.load(train_server_path)
    else:
        train_data = datasets.MNIST(root=path, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),]))
        test_data = datasets.MNIST(root=path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),]))
        num_classes = len(train_data.classes)
        server_number_sampled = dataset_configs["server_train_number"]//num_classes
        train_server_subset = _split_by_labels(num_classes, train_data, server_number_sampled,train_server_path)
    train_server_data = TorchVisionDataset(
            name=dataset_name,
            data = train_server_subset,
            split="train",
        )

    # 1.2 Training Data for Student
    train_student_path = os.path.join(path, "train_student_split.pt")
    if os.path.exists(train_student_path):
        train_student_subset = torch.load(train_student_path)
    else:
        train_data = datasets.MNIST(root=path, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),]))
        num_classes = len(train_data.classes)
        student_number_sampled = dataset_configs["student_train_number"]//num_classes
        train_student_subset = _split_by_labels(num_classes, train_data, student_number_sampled,train_student_path)
    train_student_data = TorchVisionDataset(
            name=dataset_name,
            data = train_student_subset,
            split="train",
        )
    
    # 2.1 Test Data for Server
    test_server_path = os.path.join(path, "test_server_split.pt")
    if os.path.exists(test_server_path):
        test_server_subset = torch.load(test_server_path)
    else:
        test_data = datasets.MNIST(root=path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),]))
        num_classes = len(test_data.classes)
        server_number_sampled = dataset_configs["server_test_number"]//num_classes
        test_server_subset = _split_by_labels(num_classes, test_data, server_number_sampled,test_server_path)
    test_server_data = TorchVisionDataset(
            name=dataset_name,
            data = test_server_subset,
            split="test",
        )

    # 2.2 Test Data for Student
    test_student_path = os.path.join(path, "test_student_split.pt")
    if os.path.exists(test_student_path):
        test_student_subset = torch.load(test_student_path)
    else:
        test_data = datasets.MNIST(root=path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),]))
        num_classes = len(test_data.classes)
        student_number_sampled = dataset_configs["student_test_number"]//num_classes
        test_student_subset = _split_by_labels(num_classes, test_data, student_number_sampled,test_student_path)
    test_student_data = TorchVisionDataset(
            name=dataset_name,
            data = test_student_subset,
            split="test",
        )
    # train_server_subset_dataset = torch.utils.data.TensorDataset(train_server_subset)
    print(f"train_server_data length: {len(train_server_data)}, train_student_data length: {len(train_student_data)}, test_server_data length: {len(test_server_data)}, test_student_data length: {len(test_student_data)}")

    return {"train": train_server_data, "test": test_student_data}


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

