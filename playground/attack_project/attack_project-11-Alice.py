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
        l2_threshold=7.5

    ):
        self.vm = vm
        self.image_size = image_size
        self.l2_threshold = l2_threshold

    def attack(
        self,
        original_image:  np.ndarray,
        labels: List[int],
        target_label: int,
        epsilon = 0.214

    ):
        """
        currently this attack has 2 versions, 1 with no mask pre-defined, 1 with mask pre-defined.
        args:
            original_image: a numpy ndarray images, [1,28,28]
            labels: label of the image, a list of size 1
            target_label: target label we want the image to be classified, int
        return:
            the perturbed image
            label of that perturbed iamge
            success: whether the attack succeds
        """

        perturbed_image = deepcopy(original_image)
        # --------------TODO--------------
        data_grad = None
        sign_data_grad = None
        perturbed_image = None
        # ------------END TODO-------------
        return perturbed_image.cpu().detach().numpy()


def main():
    # (1) prepare the data loaders and the model
    server_url = "http://128.195.151.199:5000"  # used when the student needs to debug on the server
    local_url = "http://127.0.0.1:5000"  # used when the student needs to debug locally

    vm = virtual_model(local_url, application_name="Project_Attack")
    target_label = 7
    dev_data = get_data(application="Project_Attack", data_type="test")

    targeted_dev_data = []
    for instance in dev_data:
        if instance["label"] != target_label:
            targeted_dev_data.append(instance)
    print(len(targeted_dev_data))
    targeted_dev_data = targeted_dev_data
    universal_perturb_batch_size = 1
    # tokenizer = model_wrapper.get_tokenizer()
    iterator_dataloader = DataLoader(
        targeted_dev_data,
        batch_size=universal_perturb_batch_size,
        shuffle=True,
        collate_fn=default_data_collator,
    )
    print("started the process")
    all_vals = []
    correct = 0
    adv_examples = []
    print("start testing")
    # Loop over all examples in test set
    test_loader = iterator_dataloader
    epsilon = 0.214
    method = ProjectAttack(vm, image_size=[1, 3, 32, 32])
    for batch in test_loader:
        # Call FGSM Attack
        labels = batch["labels"]
        # print(batch)
        perturbed_data = method.attack(
            batch["image"].cpu().detach().numpy(),
            labels.cpu().detach().numpy(),
            [target_label],
            epsilon=epsilon,
        )

        # Re-classify the perturbed image
        output = vm.get_batch_output(perturbed_data, labels.cpu().detach().numpy(),)
        final_pred = np.argmax(output[0])
        # final_pred = output.max(1, keepdim=True)[1]
        # print(output, final_pred)
        if final_pred.item() != labels.item():
            correct += 1

    # Calculate final accuracy for this epsilon
    final_acc = correct / float(len(test_loader))
    print(
        "Epsilon: {}\tTest Accuracy = {} / {} = {}".format(
            epsilon, correct, len(test_loader), final_acc
        )
    )


if __name__ == "__main__":
    main()