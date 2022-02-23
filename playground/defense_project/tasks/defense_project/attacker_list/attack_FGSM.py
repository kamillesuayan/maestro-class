from typing import List, Iterator, Dict, Tuple, Any, Type
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import DataLoader
from transformers.data.data_collator import default_data_collator
import torch.nn.functional as F

class Attack:
    def __init__(self, model, device, epsilon=0.2, min_val=0, max_val=1):
        self.model = model.to(device)
        self.device = device
        self.epsilon = epsilon
        self.min_val = min_val
        self.max_val = max_val

    def attack(self, original_images, labels):
        original_images = original_images.clone().detach().to(self.device)
        original_images = torch.unsqueeze(original_images, 0)
        labels = torch.tensor(labels).to(self.device)
        original_images.requires_grad = True
        self.model.eval()
        outputs = self.model(original_images)
        loss = F.nll_loss(outputs, labels)
        self.model.zero_grad()
        loss.backward()
        data_grad = original_images.grad.data
        sign_data_grad = data_grad.sign()
        perturbed_image = original_images + self.epsilon*sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, self.min_val, self.max_val)

        adv_outputs = self.model(perturbed_image)
        final_pred = adv_outputs.max(1, keepdim=True)[1]
        correct = 0
        if final_pred.item() != labels.item():
            correct = 1
        return perturbed_image.cpu().detach().numpy(), correct

