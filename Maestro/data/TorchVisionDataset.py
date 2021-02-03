from numpy.lib.polynomial import RankWarning
from textattack.datasets import TextAttackDataset
from torchvision import datasets
import collections

import random


class TorchVisionDataset(TextAttackDataset):
    """
    - name: the dataset name
    - subset: the subset of the main dataset. Dataset will be loaded as ``nlp.load_dataset(name, subset)``.
    - label_map: Mapping if output labels should be re-mapped. Useful
      if model was trained with a different label arrangement than
      provided in the ``nlp`` version of the dataset.
    - output_scale_factor (float): Factor to divide ground-truth outputs by.
        Generally, TextAttack goal functions require model outputs
        between 0 and 1. Some datasets test the model's correlation
        with ground-truth output, instead of its accuracy, so these
        outputs may be scaled arbitrarily.
    - shuffle (bool): Whether to shuffle the dataset on load.
    """

    def __init__(
        self, name, split="train", transforms=None, shuffle=False,
    ):
        self._name = name
        self._dataset = datasets.MNIST(
            "../tmp", train=False, download=True, transform=transforms,
        )

        # Input/output column order, like (('premise', 'hypothesis'), 'label')

        self.input_columns, self.output_column = ("image", "label")

        self._i = 0
        self.examples = list(self._dataset)

        if shuffle:
            random.shuffle(self.examples)

    def _format_raw_example(self, raw_example):
        # input_dict = collections.OrderedDict(
        #     [(c, raw_example[c]) for c in self.input_columns]
        # )

        # output = raw_example[self.output_column]
        # if self.label_map:
        #     output = self.label_map[output]
        # if self.output_scale_factor:
        #     output = output / self.output_scale_factor

        # return (input_dict, output)

        return raw_example

    def __next__(self):
        if self._i >= len(self.examples):
            raise StopIteration
        raw_example = self.examples[self._i]
        self._i += 1
        return self._format_raw_example(raw_example)

    def __getitem__(self, i):
        if isinstance(i, int):
            return self._format_raw_example(self.examples[i])
        else:
            # `i` could be a slice or an integer. if it's a slice,
            # return the formatted version of the proper slice of the list
            return [self._format_raw_example(ex) for ex in self.examples[i]]

    def get_json_data(self):
        if self.examples:
            new_data = []
            for idx, instance in enumerate(self.examples):
                new_instance = {}
                new_instance["image"] = instance[0].numpy().tolist()
                new_instance["label"] = instance[1]
                new_instance["uid"] = idx
                new_data.append(new_instance)
        return new_data
