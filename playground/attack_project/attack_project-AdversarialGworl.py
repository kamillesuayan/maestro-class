from typing import List, Iterator, Dict, Tuple, Any, Type
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import DataLoader
from Maestro.attacker_helper.attacker_request_helper import virtual_model
from transformers.data.data_collator import default_data_collator
from Maestro.evaluator.Evaluator import get_data

class ProjectAttack:
    def __init__(
        self,
        vm,
        image_size: List[int],
        l2_threshold=7.5,
        steps=3, # strong correlation with distance??
        alpha=0.019 # tune
        
    ):
        self.vm = vm
        self.image_size = image_size
        self.l2_threshold = l2_threshold
        self.steps = steps
        self.alpha = alpha
        

    def attack(
        self,
        original_image:  np.ndarray,
        labels: List[int],
        target_label: int,
        epsilon = 0.214 # tune

    ):
        """
        args:
            original_image: a numpy ndarray images, [1,3,32,32]
            labels: label of the image, a list of size 1
            target_label: target label we want the image to be classified, int
        return:
            the perturbed image
            label of that perturbed iamge
            success: whether the attack succeds
        """

        perturbed_image = deepcopy(original_image)
        # --------------TODO--------------
        # To run this file change the group_name(#L:41) in attack_project-Evaluation.py to "FGSM_Targeted"

        target_labels = [target_label]*len(labels)
        for i in range(self.steps):
            # Calculate the gradient of loss with respect to the image and target_label
            data_grad = self.vm.get_batch_input_gradient(perturbed_image, target_labels)
            data_grad = torch.FloatTensor(data_grad)

            # Determine the direction of gradient using sign()
            sign_data_grad = data_grad.sign()

            # Perturb the image in the direction of gradient with respect to target_label by epsilon
            # Ensure that it is TARGETED

            # grad = self.alpha*(torch.FloatTensor(original_image) - epsilon * sign_data_grad)
            # perturbed_image = original_image - grad
            # np_perturbed = perturbed_image.cpu().detach().numpy()
            # grad = grad.cpu().detach().numpy()
            # step_grad = torch.clamp(grad, -epsilon, epsilon)

            # np_perturbed = np.clip(perturbed_image, perturbed_image, grad)
            # perturbed_image = original_image + grad

            
            # perturbed_image = np.clip(np_perturbed, self.x_p, np_perturbed)

            temp_grad = self.alpha * sign_data_grad
            t = torch.from_numpy(original_image) + temp_grad

            final_grad = t - perturbed_image
            final_grad = torch.clamp(final_grad, -epsilon, epsilon)
            x_p = perturbed_image - final_grad.cpu().detach().numpy()

            # Clip the value of each pixel to be between 0 & 1
            perturbed_image = np.clip(x_p, 0,1)

        # ------------END TODO-------------

        return x_p

