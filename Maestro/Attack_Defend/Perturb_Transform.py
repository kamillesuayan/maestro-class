from typing import List, Iterator, Dict, Tuple, Any, Type
import numpy as np
import torch
from copy import deepcopy
from Maestro.attacker_helper.attacker_request_helper import virtual_model
from tqdm import tqdm


class FGSMAttack:
    def __init__(
        self, app, epsilon: float = 0.2,
    ):
        self.app = app
        self.epsilon = epsilon

    def attack(
        self, original_image: List[List[int]], labels: List[int],
    ):
        perturbed_image = deepcopy(original_image)
        data_grad = (
            self.app.get_batch_input_gradient(perturbed_image, labels).cpu().detach()
        )
        data_grad = torch.FloatTensor(data_grad)
        sign_data_grad = data_grad.sign()
        perturbed_image = (
            torch.FloatTensor(original_image) + self.epsilon * sign_data_grad
        )
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        return perturbed_image.cpu().detach()


PERTURB_METHODS = {"FGSM": FGSMAttack, "PGD": None}


def perturb_transform(app, data, perturb_name):
    print("begin perturbing")
    perturb = PERTURB_METHODS[perturb_name](app)
    new_data = []
    for instance in tqdm(data):
        x = [instance[0].numpy().tolist()]
        label = [instance[1]]
        # print(x, label)
        perturbed_x = perturb.attack(x, label)
        new_data.append((perturbed_x, label[0]))
    data.examples = new_data
    return data

