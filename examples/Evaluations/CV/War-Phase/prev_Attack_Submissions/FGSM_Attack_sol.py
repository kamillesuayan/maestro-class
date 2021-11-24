from typing import List, Iterator, Dict, Tuple, Any, Type
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import DataLoader
from Maestro.attacker_helper.attacker_request_helper import virtual_model
from transformers.data.data_collator import default_data_collator


class FGSMAttack:
    def __init__(self):
        pass

    def attack(
        original_image: List[List[int]],
        labels: List[int],
        vm: virtual_model,
        epsilon: float,
    ):
        perturbed_image = deepcopy(original_image)
        # --------------TODO--------------
        data_grad = vm.get_batch_input_gradient(perturbed_image, labels)
        data_grad = torch.FloatTensor(data_grad)
        sign_data_grad = data_grad.sign()
        perturbed_image = torch.FloatTensor(original_image) + epsilon * sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # ------------END TODO-------------
        return perturbed_image.cpu().detach().numpy()


# def main():
#     # (1) prepare the data loaders and the model
#     server_url = "http://128.195.56.136:5000"  # used when the student needs to debug on the server
#     local_url = "http://127.0.0.1:5000"  # used when the student needs to debug locally

#     vm = virtual_model(local_url, application_name="FGSM")
#     dataset_label_filter = 0
#     dev_data = vm.get_data(data_type="test")

#     targeted_dev_data = []
#     for instance in dev_data:
#         if instance["label"] == dataset_label_filter:
#             targeted_dev_data.append(instance)
#     print(len(targeted_dev_data))
#     targeted_dev_data = targeted_dev_data[:10]
#     universal_perturb_batch_size = 1
#     # tokenizer = model_wrapper.get_tokenizer()
#     iterator_dataloader = DataLoader(
#         targeted_dev_data,
#         batch_size=universal_perturb_batch_size,
#         shuffle=True,
#         collate_fn=default_data_collator,
#     )
#     print("started the process")
#     all_vals = []
#     correct = 0
#     adv_examples = []
#     print("start testing")
#     # Loop over all examples in test set
#     test_loader = iterator_dataloader
#     epsilon = 0.2
#     for batch in test_loader:
#         # Call FGSM Attack
#         labels = batch["labels"]
#         # print(batch)
#         perturbed_data = attack(
#             batch["image"].cpu().detach().numpy(),
#             labels.cpu().detach().numpy(),
#             vm,
#             epsilon=epsilon,
#         )

#         # Re-classify the perturbed image
#         output = vm.get_batch_output(perturbed_data, labels)
#         final_pred = np.argmax(output[0])
#         # final_pred = output.max(1, keepdim=True)[1]
#         # print(output, final_pred)
#         if final_pred.item() == labels.item():
#             correct += 1

#     # Calculate final accuracy for this epsilon
#     final_acc = correct / float(len(test_loader))
#     print(
#         "Epsilon: {}\tTest Accuracy = {} / {} = {}".format(
#             epsilon, correct, len(test_loader), final_acc
#         )
#     )


# if __name__ == "__main__":
#     main()
