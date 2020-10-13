import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import textattack
from textattack.datasets import HuggingFaceDataset
from Maestro.data.process_data import prepare_dataset_for_training
import numpy as np
from torch.utils.data import DataLoader, RandomSampler

# from allennlp.data.vocabulary import Vocabulary
# from allennlp.data.token_indexers import SingleIdTokenIndexer
# from allennlp.data.samplers import BucketBatchSampler
# from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import (
#     StanfordSentimentTreeBankDatasetReader,
# )

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
        return read_sst()
        # single_id_indexer = SingleIdTokenIndexer(
        #     lowercase_tokens=True
        # )  # word tokenizer
        # # use_subtrees gives us a bit of extra data by breaking down each example into sub sentences.
        # reader = StanfordSentimentTreeBankDatasetReader(
        #     granularity="2-class",
        #     token_indexers={"tokens": single_id_indexer},
        #     use_subtrees=False,
        # )
        # train_data = reader.read(train_data_path)
        # reader = StanfordSentimentTreeBankDatasetReader(
        #     granularity="2-class", token_indexers={"tokens": single_id_indexer}
        # )
        # dev_data = reader.read(dev_data_path)
        # test_data = reader.read(test_data_path)
        # vocab = Vocabulary.from_instances(train_data)
        # train_data.index_with(vocab)
        # dev_data.index_with(vocab)
        # train_sampler = BucketBatchSampler(
        #     train_data, batch_size=32, sorting_keys=["tokens"]
        # )
        # validation_sampler = BucketBatchSampler(
        #     dev_data, batch_size=32, sorting_keys=["tokens"]
        # )
        # test_sampler = BucketBatchSampler(
        #     test_data, batch_size=32, sorting_keys=["tokens"]
        # )

        # train_loader = DataLoader(train_data, batch_sampler=train_sampler)
        # dev_loader = DataLoader(dev_data, batch_sampler=validation_sampler)

        # test_loader = DataLoader(test_data, batch_sampler=test_sampler)

    return train_loader, dev_loader, test_loader, vocab


def read_sst():
    train_data = HuggingFaceDataset(
        name="glue", subset="sst2", split="train", label_map=None, shuffle=True
    )
    train_data, train_label = prepare_dataset_for_training(train_data)
    dev_data = HuggingFaceDataset(
        name="glue", subset="sst2", split="test", label_map=None, shuffle=True
    )
    dev_data, dev_label = prepare_dataset_for_training(dev_data)

    return train_data, train_label, dev_data, dev_label


def batch_encode(tokenizer, text_list):
    if hasattr(tokenizer, "batch_encode"):
        return tokenizer.batch_encode(text_list)
    else:
        return [tokenizer.encode(text_input) for text_input in text_list]


def _make_dataloader(tokenizer, text, labels, batch_size):
    """Create torch DataLoader from list of input text and labels.
    :param tokenizer: Tokenizer to use for this text.
    :param text: list of input text.
    :param labels: list of corresponding labels.
    :param batch_size: batch size (int).
    :return: torch DataLoader for this training set.
    """
    text_ids = batch_encode(tokenizer, text)
    input_ids = np.array(text_ids)
    labels = np.array(labels)
    data = list((ids, label) for ids, label in zip(input_ids, labels))
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return dataloader
