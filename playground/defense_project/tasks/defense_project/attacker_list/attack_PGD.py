from typing import List, Iterator, Dict, Tuple, Any, Type
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import DataLoader
from transformers.data.data_collator import default_data_collator
import torch.nn.functional as F

class Attack:
    # PGD Attack
    def __init__(self, model, device, epsilon=0.2, alpha=0.1, min_val=0, max_val=1, max_iters=10,  _type='linf'):
        self.model = model.to(device)
        self.device = device
        self.epsilon = epsilon
        self.min_val = min_val
        self.max_val = max_val
        self.alpha = alpha
        self.max_iters = max_iters
        self._type = _type

    def project(self, x, original_x, epsilon, _type='linf'):
        if _type == 'linf':
            max_x = original_x + epsilon
            min_x = original_x - epsilon
            x = torch.max(torch.min(x, max_x), min_x)
        elif _type == 'l2':
            dist = (x - original_x)
            dist = dist.view(x.shape[0], -1)
            dist_norm = torch.norm(dist, dim=1, keepdim=True)
            mask = (dist_norm > epsilon).unsqueeze(2).unsqueeze(3)
            # dist = F.normalize(dist, p=2, dim=1)
            dist = dist / dist_norm
            dist *= epsilon
            dist = dist.view(x.shape)
            x = (original_x + dist) * mask.float() + x * (1 - mask.float())
        else:
            raise NotImplementedError
        return x

    def attack(self, original_images, labels, reduction4loss='mean', random_start=False):
        original_images = torch.unsqueeze(original_images, 0).to(self.device)
        labels = torch.tensor(labels).to(self.device)

        if random_start:
            rand_perturb = torch.FloatTensor(original_images.shape).uniform_(
                -self.epsilon, self.epsilon)
            rand_perturb = rand_perturb.to(self.device)
            x = original_images + rand_perturb
            x.clamp_(self.min_val, self.max_val)
        else:
            x = original_images.clone()

        x.requires_grad = True

        # max_x = original_images + self.epsilon
        # min_x = original_images - self.epsilon

        with torch.enable_grad():
            for _iter in range(self.max_iters):
                self.model.eval()
                # print(x.shape)
                outputs = self.model(x)

                loss = F.cross_entropy(outputs, labels, reduction=reduction4loss)

                if reduction4loss == 'none':
                    grad_outputs = torch.ones(loss.shape).to(self.device)

                else:
                    grad_outputs = None

                grads = torch.autograd.grad(loss, x, grad_outputs=grad_outputs,
                        only_inputs=True)[0]

                x.data += self.alpha * torch.sign(grads.data)
                x = self.project(x, original_images, self.epsilon, self._type)
                x.clamp_(self.min_val, self.max_val)

        final_pred = outputs.max(1, keepdim=True)[1]

        correct = 0
        if final_pred.item() != labels.item():
            correct = 1
        # return x
        return x.cpu().detach().numpy(), correct

