from torch.utils.data import DataLoader

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

