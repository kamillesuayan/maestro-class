from typing import List, Iterator, Dict, Tuple, Any, Type
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers.data.data_collator import default_data_collator

# ------------------ LOCAL IMPORTS ---------------------------------
from Maestro.defense_helper.defense_request_helper import virtual_model
# ------------------ LOCAL IMPORTS ---------------------------------

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



def detector():
    # --------------TODO--------------



    # ------------END TODO-------------
    return model
