import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import (
    StanfordSentimentTreeBankDatasetReader,
)

WRITE_ACESS = 1
READ_ACESS = 2
READ_WRITE_ACESS = 3


class DataModifier:
    """
    """

    def __init__(self, data: DataLoader, access: int) -> None:
        self.data = data
        self.access = access

    def get_read_data(self):
        if self.access != 3 and self.access != 1:
            raise ValueError("Do not have access to read data")
        return self.data

    def get_write_data(self):
        if self.access != 3 and self.access != 2:
            raise ValueError("Do not have access to write data")
        return self.data

    # def read(self, idx):
    #     if self.access != 3 and self.access != 1:
    #         raise ValueError("Do not have access to read data")
    #     return self.data[idx]

    # def write(self, type, idxes, values):
    #     if self.access != 3 and self.access != 2:
    #         raise ValueError("Do not have access to write data")
    #     if type == "front":
    #         self.data[idxes] = torch.cat(values, self.data[idxes])
    #     elif type == "back":
    #         self.data[idxes] = torch.cat(self.data[idxes], values)


def get_data(name: str):
    vocab = None
    if name == "MNIST":
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "../tmp",
                train=False,
                download=True,
                transform=transforms.Compose([transforms.ToTensor(),]),
            ),
            batch_size=1,
            shuffle=True,
        )
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "../tmp",
                train=True,
                download=True,
                transform=transforms.Compose([transforms.ToTensor(),]),
            ),
            batch_size=1,
            shuffle=True,
        )
        dev_loader = None
    elif name == "SST":
        single_id_indexer = SingleIdTokenIndexer(
            lowercase_tokens=True
        )  # word tokenizer
        # use_subtrees gives us a bit of extra data by breaking down each example into sub sentences.
        reader = StanfordSentimentTreeBankDatasetReader(
            granularity="2-class",
            token_indexers={"tokens": single_id_indexer},
            use_subtrees=False,
        )
        train_data = reader.read(train_data_path)
        reader = StanfordSentimentTreeBankDatasetReader(
            granularity="2-class", token_indexers={"tokens": single_id_indexer}
        )
        dev_data = reader.read(dev_data_path)
        test_data = reader.read(test_data_path)
        vocab = Vocabulary.from_instances(train_data)
        train_data.index_with(vocab)
        dev_data.index_with(vocab)
        train_sampler = BucketBatchSampler(
            train_data, batch_size=32, sorting_keys=["tokens"]
        )
        validation_sampler = BucketBatchSampler(
            dev_data, batch_size=32, sorting_keys=["tokens"]
        )
        test_sampler = BucketBatchSampler(
            test_data, batch_size=32, sorting_keys=["tokens"]
        )

        train_loader = DataLoader(train_data, batch_sampler=train_sampler)
        dev_loader = DataLoader(dev_data, batch_sampler=validation_sampler)

        test_loader = DataLoader(test_data, batch_sampler=test_sampler)

    return train_loader, dev_loader, test_loader, vocab


def read_sst():
    train_data_path = (
        "https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/train.txt"
    )
    dev_data_path = "https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/dev.txt"
    test_data_path = "https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/test.txt"

