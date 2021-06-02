from typing import List, Iterator, Dict, Tuple, Any, Type
import numpy as np
import torch
from torch.utils.data import DataLoader
from attacker_request_helper import virtual_model
from transformers.data.data_collator import default_data_collator


def attack(
    original_tokens: List[List[int]],
    labels: List[int],
    vm: virtual_model,
    epsilon: float,
):
    # -------------------------------- TODO ---------------------------------------
    # implement the attack for FGSM. You can either use API to take advantage of the server 
    # resources, or use local resources via PyTorch
    # ---------------------------------TODO END-----------------------------------------
    return perturbed_image


def main():
    # (1) prepare the data loaders and the model
    url = "http://128.195.56.136:5000"

    vm = virtual_model(url, application_name="FGSM")
    dataset_label_filter = 0
    dev_data = vm.get_data(data_type="test")

    targeted_dev_data = []
    for instance in dev_data:
        if instance["label"] == dataset_label_filter:
            targeted_dev_data.append(instance)
    print(len(targeted_dev_data))
    targeted_dev_data = targeted_dev_data[:10]
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
    epsilon = 0.01
    for batch in test_loader:
        # Call FGSM Attack
        labels = batch["labels"]
        print(batch)
        perturbed_data = attack(
            batch["image"].cpu().detach().numpy(),
            labels.cpu().detach().numpy(),
            vm,
            epsilon=epsilon,
        )

        # Re-classify the perturbed image
        output = vm.get_output(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    # Calculate final accuracy for this epsilon
    final_acc = correct / float(len(test_loader))
    print(
        "Epsilon: {}\tTest Accuracy = {} / {} = {}".format(
            epsilon, correct, len(test_loader), final_acc
        )
    )


if __name__ == "__main__":
    main()
