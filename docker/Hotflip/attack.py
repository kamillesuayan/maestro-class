
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
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils.data.sampler import BatchSampler, RandomSampler
# from transformers import (
#   Trainer,
#   TrainingArguments,
#   EvalPrediction,
#   glue_compute_metrics,
# )
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

def hotflip_attack(original_tokens:List[List[int]], labels:List[int],vm:virtual_model,constraint:int):
    '''
        The function that the student/attacker needs to implement.
        Given:
            original_tokens: input tokens 
            constraint: the budget, e.g., <= 10 tokens.
        Output:
            perturbed_tokens：perturbed tokens that increase the overall loss
    '''
    flipped = []
    print(original_tokens,original_tokens.shape)
    perturbed_tokens = copy.deepcopy(original_tokens)
    for i in range(constraint):
    # -------- TODO --------
        data_grad = vm.get_batch_input_gradient(perturbed_tokens,labels)
        # print(len(data_grad))
        # data_grad of shape [B, T, D] e.g., [1,128,768]
        data_grad = data_grad[0]
        # print(len(data_grad))
        # print(len(data_grad[0]))
        grads_magnitude = [np.dot(g,g) for g in data_grad]
        print(grads_magnitude)
        # only flip a token once
        for index in flipped:
            grads_magnitude[index] = -1

        # We flip the token with highest gradient norm.
        index_of_token_to_flip = np.argmax(grads_magnitude)
        print("index of token to flip: {}".format(index_of_token_to_flip))
        if grads_magnitude[index_of_token_to_flip] == -1:
            # If we've already flipped all of the tokens, we give up.
            break
        flipped.append(index_of_token_to_flip)
        # index_of_token_to_flip = 1
        embedding_weight = vm.get_embedding()
        grad = data_grad[index_of_token_to_flip]
        cand_id = hotflip_attack_helper(
            grad,
            embedding_weight,
            num_candidates=1,
            increase_loss=True,
        )
        print("cand ids:", cand_id)
        perturbed_tokens[0][index_of_token_to_flip] = cand_id[0]
        logits = vm.get_batch_output(perturbed_tokens,labels)
        print(logits)
        # print("og label", batch["labels"].cpu().detach().numpy()[0])
        # print("oglogits: {}, logits: {}".format(oglogits[0], logits[0]))
    preds = np.argmax(logits[1], axis=1)
    term = preds != labels
    if term:
        print("attack success")
    else:
        print("attack fail")
    # -------- TODO END--------
    return perturbed_tokens,term



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
    # url = "http://127.0.0.1:5000"
    url = "http://128.195.56.136:5000"

    # payload = {
    #     "Application_Name": "Universal_Attack",
    #     "data":[]
    # }
    # final_url = url + "/get_batch_output"
    # # print(json.dumps(payload))
    # # headers = {"Content-type": "multipart/form-data"}
    # # headers = {"Content-type": "application/json", "Accept": "text/plain"}
    # response = requests.post(
    #     final_url, data=payload
    # )
    # print(response.json())
    vm = virtual_model(url)
    dataset_label_filter = 0
    dev_data = vm.get_data()

    targeted_dev_data = []
    for instance in dev_data:
        if instance["label"] == dataset_label_filter:
            targeted_dev_data.append(instance)
    print(len(targeted_dev_data))
    targeted_dev_data = targeted_dev_data[:1]
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
    indexes = []
    cand_ids = []
    # og_acc = get_accuracy(
    #   model_wrapper, dev_data, tokenizer, indexes, cand_ids, batch=False
    # )
    new_batches = []
    for batch in iterator_dataloader:
        print("start_batch")
        # print(batch)
        flipped = [0]
        # acc, oglogits = get_accuracy(
        #   model_wrapper, batch, tokenizer, indexes, cand_ids, batch=True
        # )
        indexes = []
        cand_ids = []
        # get gradient w.r.t. trigger embeddings for current batch
        labels = batch["labels"]
        perturbed,success = hotflip_attack(batch["input_ids"].cpu().detach().numpy(),labels.cpu().detach().numpy(),vm,constraint=1)

        all_vals.append(success)
        
    print("After attack attack success rate")
    a = np.array(all_vals).mean()
    print(a)



if __name__ == "__main__":
    main()
