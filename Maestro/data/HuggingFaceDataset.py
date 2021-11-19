import collections
import random

import datasets

# import textattack
import numpy as np
from torch.utils.data import DataLoader, RandomSampler
import torch

# from textattack.shared import AttackedText
def prepare_dataset_for_training(nlp_dataset, _format_raw_example):
    """
        Copied from textattack
    """
    """Changes an `nlp` dataset into the proper format for tokenization."""

    def prepare_example_dict(ex):
        """Returns the values in order corresponding to the data.
        ex:
            'Some text input'
        or in the case of multi-sequence inputs:
            ('The premise', 'the hypothesis',)
        etc.
        """
        values = list(ex.values())
        if len(values) == 1:
            return values[0]
        return tuple(values)

    # nlp_dataset._i = 0

    text, outputs = zip(*((prepare_example_dict(x[0]), x[1]) for x in nlp_dataset))
    return list(text), list(outputs)


def _batch_encode(tokenizer, text_list, max_length):

    # if hasattr(tokenizer, "batch_encode"):

    return tokenizer.batch_encode_plus(
        text_list,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
        padding="max_length",
    )


# else:

#     return [tokenizer.encode(text_input) for text_input in text_list]


def make_text_dataloader(tokenizer, data, batch_size):
    """
    Copied from textattack
    Create torch DataLoader from list of input text and labels.
    :param tokenizer: Tokenizer to use for this text.
    :param text: list of input text.
    :param labels: list of corresponding labels.
    :param batch_size: batch size (int).
    :return: torch DataLoader for this training set.
    """
    data, labels = prepare_dataset_for_training(data)
    text_ids = _batch_encode(tokenizer, data)
    input_ids = np.array(text_ids)
    labels = np.array(labels)
    data = list((ids, label) for ids, label in zip(input_ids, labels))
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return dataloader


def _cb(s):
    """Colors some text blue for printing to the terminal."""
    return textattack.shared.utils.color_text(str(s), color="blue", method="ansi")


def whether_two_inputs(dataset):
    input_columns, _ = get_datasets_dataset_columns(dataset)
    return True if (input_columns == 2) else False


def get_datasets_dataset_columns(dataset):
    # referenced from textattack
    schema = set(dataset.column_names)
    if {"premise", "hypothesis", "label"} <= schema:
        input_columns = ("premise", "hypothesis")
        output_column = "label"
    elif {"question", "sentence", "label"} <= schema:
        input_columns = ("question", "sentence")
        output_column = "label"
    elif {"sentence1", "sentence2", "label"} <= schema:
        input_columns = ("sentence1", "sentence2")
        output_column = "label"
    elif {"question1", "question2", "label"} <= schema:
        input_columns = ("question1", "question2")
        output_column = "label"
    elif {"question", "sentence", "label"} <= schema:
        input_columns = ("question", "sentence")
        output_column = "label"
    elif {"text", "label"} <= schema:
        input_columns = ("text",)
        output_column = "label"
    elif {"sentence", "label"} <= schema:
        input_columns = ("sentence",)
        output_column = "label"
    elif {"document", "summary"} <= schema:
        input_columns = ("document",)
        output_column = "summary"
    elif {"content", "summary"} <= schema:
        input_columns = ("content",)
        output_column = "summary"
    elif {"label", "review"} <= schema:
        input_columns = ("review",)
        output_column = "label"
    else:
        raise ValueError(
            f"Unsupported dataset schema {schema}. Try loading dataset manually (from a file) instead."
        )

    return input_columns, output_column


class HuggingFaceDataset:
    """Loads a dataset from HuggingFace ``datasets`` and prepares it as a
    TextAttack dataset.

    - name: the dataset name
    - subset: the subset of the main dataset. Dataset will be loaded as ``datasets.load_dataset(name, subset)``.
    - label_map: Mapping if output labels should be re-mapped. Useful
      if model was trained with a different label arrangement than
      provided in the ``datasets`` version of the dataset.
    - output_scale_factor (float): Factor to divide ground-truth outputs by.
        Generally, TextAttack goal functions require model outputs
        between 0 and 1. Some datasets test the model's correlation
        with ground-truth output, instead of its accuracy, so these
        outputs may be scaled arbitrarily.
    - shuffle (bool): Whether to shuffle the dataset on load.
    """

    def __init__(
        self,
        name,
        subset=None,
        split="train",
        label_map=None,
        output_scale_factor=None,
        dataset_columns=None,
        shuffle=False,
    ):
        self._name = name
        if name != None:
            self._dataset = datasets.load_dataset(name, subset)
            self._dataset = self._dataset[split]
        else:
            self._dataset = None
        # subset_print_str = f", subset {_cb(subset)}" if subset else ""
        # textattack.shared.logger.info(
        #     f"Loading {_cb('datasets')} dataset {_cb(name)}{subset_print_str}, split {_cb(split)}."
        # )
        # Input/output column order, like (('premise', 'hypothesis'), 'label')
        self.dataset_columns = dataset_columns
        (
            self.input_columns,
            self.output_column,
        ) = dataset_columns or get_datasets_dataset_columns(self._dataset)
        self._i = 0
        self.label_map = label_map
        self.output_scale_factor = output_scale_factor
        try:
            self.label_names = self._dataset.features["label"].names
            # If labels are remapped, the label names have to be remapped as
            # well.
            if label_map:
                self.label_names = [
                    self.label_names[self.label_map[i]]
                    for i in range(len(self.label_map))
                ]
        except KeyError:
            # This happens when the dataset doesn't have 'features' or a 'label' column.
            self.label_names = None
        except AttributeError:
            # This happens when self._dataset.features["label"] exists
            # but is a single value.
            self.label_names = ("label",)

        self.examples = list(self._dataset)
        for idx, ex in enumerate(self.examples):
            self.examples[idx] = self._format_raw_example(ex)
        self.examples_indexed = None
        if shuffle:
            random.shuffle(self.examples)

    def __next__(self):
        if self._i >= len(self.examples_indexed):
            raise StopIteration
        if self.examples_indexed is None:
            raw_example = self.examples[self._i]
            self._i += 1
            return raw_example
        else:
            raw_example = self.examples_indexed[self._i]
            self._i += 1
            return raw_example

    def __getitem__(self, i):
        if self.examples_indexed is None:
            if isinstance(i, int):
                return self.examples[i]
            elif isinstance(i, list) or isinstance(i, torch.Tensor):
                return [self.examples[ex] for ex in i]
            else:
                return [ex for ex in self.examples[i]]
        else:
            if isinstance(i, int):
                return self.examples_indexed[i]
            elif isinstance(i, list) or isinstance(i, torch.Tensor):
                return [self.examples_indexed[ex] for ex in i]
            else:
                return [ex for ex in self.examples_indexed[i]]

    def _format_raw_example(self, raw_example):
        input_dict = collections.OrderedDict(
            [(c, raw_example[c]) for c in self.input_columns]
        )
        output = raw_example[self.output_column]
        if self.label_map:
            output = self.label_map[output]
        if self.output_scale_factor:
            output = output / self.output_scale_factor

        return (input_dict, output)

    def indexed(self, tokenizer, max_length, uid=True):
        input_columns, output_column = get_datasets_dataset_columns(self._dataset)
        if whether_two_inputs(self._dataset):
            data = self._dataset.map(
                lambda e: tokenizer(
                    e["premise"],
                    e["hypothesis"],
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                ),
                batched=True,
            )
            data.set_format(
                type="torch",
                columns=["input_ids", "token_type_ids", "attention_mask", "label"],
            )
        else:
            data = self._dataset.map(
                lambda e: tokenizer(
                    e[input_columns[0]],
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                ),
                batched=True,
            )
            data.set_format(
                type="torch",
                columns=["input_ids", "token_type_ids", "attention_mask", "label"],
            )
        # for i in range(len(self.examples)):
        #     inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        #     inputs["labels"] = labels[i]
        #     data.append(inputs)
        indexed_data = data
        if uid:
            indexed_data = []
            for i, dict_instance in enumerate(data):
                dict_instance["uid"] = i
                indexed_data.append(dict_instance)
            self.examples_indexed = indexed_data
        return indexed_data

    def make_text_dataloader(self, tokenizer, batch_size):
        """
        Copied from textattack
        Create torch DataLoader from list of input text and labels.
        :param tokenizer: Tokenizer to use for this text.
        :param text: list of input text.
        :param labels: list of corresponding labels.
        :param batch_size: batch size (int).
        :return: torch DataLoader for this training set.
        """
        data, labels = prepare_dataset_for_training(
            self.examples, self._format_raw_example
        )
        text_ids = _batch_encode(tokenizer, data)
        input_ids = np.array(text_ids)
        labels = np.array(labels)
        print("HuggingFaceDatset", input_ids)
        data = list((ids, label) for ids, label in zip(input_ids, labels))
        sampler = RandomSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
        return dataloader

    def get_trainable_data(self, tokenizer, max_length):
        new_dataset = self.indexed(tokenizer, max_length, False)
        return new_dataset

    def get_json_data(self):
        if self.examples_indexed:
            new_data = []
            for instance in self.examples_indexed:
                new_instance = {}
                for field in instance:
                    if isinstance(instance[field], torch.Tensor):
                        new_instance[field] = instance[field].numpy().tolist()
                    else:
                        new_instance[field] = instance[field]
                new_data.append(new_instance)
        return new_data

