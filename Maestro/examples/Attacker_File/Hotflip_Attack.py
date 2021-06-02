import os
import sys
import pickle
from flask import jsonify
import json
from copy import deepcopy
from typing import List, Iterator, Dict, Tuple, Any, Type
import numpy as np
import heapq
import requests
from operator import itemgetter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy
from attacker_request_helper import virtual_model
from transformers.data.data_collator import default_data_collator

# from Maestro.models import build_model
# from Maestro.utils import move_to_device, get_embedding
# from Maestro.data import get_dataset
# from Maestro.pipeline import (
#   AutoPipelineForNLP,
#   Pipeline,
#   Scenario,
#   Attacker,
#   model_wrapper,
# )
# from Maestro.data.HuggingFaceDataset import make_text_dataloader, HuggingFaceDataset


def attack(
    original_tokens: List[List[int]],
    labels: List[int],
    vm: virtual_model,
    constraint: int,
):
    """
        The function that the student/attacker needs to implement.
        Given:
            original_tokens: input tokens 
            constraint: the budget, e.g., <= 10 tokens.
        Output:
            perturbed_tokens：perturbed tokens that increase the overall loss
    """
    flipped = []
    perturbed_tokens = copy.deepcopy(original_tokens)
    for i in range(constraint):
    # -------------------------------- TODO ---------------------------------------
    # implement the attack for Hotflip. You can either use API to take advantage of the server 
    # resources, or use local resources via PyTorch
    # ---------------------------------TODO END-----------------------------------------
    return perturbed_tokens


def hotflip_attack_helper(
    grad, embedding_matrix, increase_loss=False, num_candidates=1,
) -> List[List[int]]:

    grad = torch.FloatTensor(grad)
    embedding_matrix = torch.FloatTensor(embedding_matrix)
    grad = grad.unsqueeze(0).unsqueeze(0)
    gradient_dot_embedding_matrix = torch.einsum(
        "bij,kj->bik", (grad, embedding_matrix)
    )
    if not increase_loss:
        gradient_dot_embedding_matrix *= (
            -1
        )  # lower versus increase the class probability.
    if num_candidates > 1:  # get top k options
        _, best_k_ids = torch.topk(gradient_dot_embedding_matrix, num_candidates, dim=2)
        return best_k_ids.detach().cpu().numpy()[0]
    _, best_at_each_step = gradient_dot_embedding_matrix.max(2)

    return best_at_each_step[0].detach().cpu().numpy()


def main():
    url = "http://128.195.56.136:5000"

    vm = virtual_model(url)
    dataset_label_filter = 0
    dev_data = vm.get_data()

    targeted_dev_data = []
    for instance in dev_data:
        if instance["label"] == dataset_label_filter:
            targeted_dev_data.append(instance)
    print(len(targeted_dev_data))
    targeted_dev_data = targeted_dev_data[:10]
    universal_perturb_batch_size = 1
    iterator_dataloader = DataLoader(
        targeted_dev_data,
        batch_size=universal_perturb_batch_size,
        shuffle=True,
        collate_fn=default_data_collator,
    )
    print("started the process")
    all_vals = []

    new_batches = []
    for batch in iterator_dataloader:
        print("start_batch")
        flipped = [0]
        labels = batch["labels"]
        perturbed, success = attack(
            batch["input_ids"].cpu().detach().numpy(),
            labels.cpu().detach().numpy(),
            vm,
            constraint=5,
        )

        all_vals.append(success)

    print("After attack attack success rate")
    a = np.array(all_vals).mean()
    print(a)


if __name__ == "__main__":
    main()
