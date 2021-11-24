from typing import List, Iterator, Dict, Tuple, Any, Type
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers.data.data_collator import default_data_collator
from copy import deepcopy

# ------------------ LOCAL IMPORTS ---------------------------------
from Maestro.attacker_helper.attacker_request_helper import virtual_model
# ------------------ LOCAL IMPORTS ---------------------------------

# class DataAugmentation:
def attack(
    original_tokens: List[List[int]],
    labels: List[int],
    vm: virtual_model,
    epsilon: float,
):
    # --------------TODO--------------
    data_grad = vm.get_batch_input_gradient(original_tokens, labels)
    data_grad = torch.FloatTensor(data_grad)
    sign_data_grad = data_grad.sign()
    perturbed_image = torch.FloatTensor(original_tokens) + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # ------------END TODO-------------
    return perturbed_image.cpu().detach().numpy()
def defense(train_dataset, vm: virtual_model, epsilon: float):
    augmented_dataset = {}
    # --------------TODO--------------
    for i in np.random.choice(np.arange(0, len(train_dataset["label"])), size=(10,)):
        perturbed_image = deepcopy(train_dataset["image"][i])
        data_grad = vm.get_batch_input_gradient(perturbed_image, labels)
        data_grad = torch.FloatTensor(data_grad)
        sign_data_grad = data_grad.sign()
        perturbed_image = torch.FloatTensor(original_image) + epsilon * sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        augmented_dataset["image"].append(perturbed_image)
        augmented_dataset["label"].append(train_dataset["label"][i])
    # ------------END TODO-------------
    return augmented_dataset
