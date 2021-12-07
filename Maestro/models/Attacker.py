from typing import List, Iterator, Dict, Tuple, Any, Type
import numpy as np
import torch


class Attacker:
    def __init__(self):
        pass

    def attack(self):
        raise NotImplementedError

    def attack_dataset(self, dataset, target_label):
        n = len(dataset)
        adversarial_examples = []
        success_list = []
        for i in range(n):
            inputs, labels = dataset[i]
            adv, success = self.attack(inputs, labels,target_label)
            adversarial_examples.append(adv)
            success_list.append(success)
        return adversarial_examples, success_list
