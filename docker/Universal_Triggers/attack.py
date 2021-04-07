
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
from torch.utils.data.sampler import RandomSampler
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

def universal_trigger_attack(iterator_dataloader:List[List[int]],dev_data,vm:virtual_model,constraint:int):
    '''
        The function that the student/attacker needs to implement.
        Given:
            original_tokens: the batch of sequences this method needs to generate triggers for 
            constraint: length of the trigger
        Output:
            perturbed_tokens：perturbed tokens that increase the overall loss
    '''
    num_trigger_tokens = constraint

    trigger_token_ids = [vm.convert_tokens_to_ids("the")] * num_trigger_tokens

    print("started the process")
    get_accuracy(vm, dev_data, trigger_token_ids, False, False)
    # best_triggers = [22775, 17950, 17087]
    # trigger_token_ids = best_triggers
    # get_accuracy(model_wrapper, dev_data, tokenizer, best_triggers, True, False)
    embedding_weight = vm.get_embedding()
    for batch in iterator_dataloader:
        # get accuracy with current triggers
        print("start_batch")
        # print(batch)
        get_accuracy(vm, batch, trigger_token_ids, False, True)
        # print(torch.cuda.memory_summary(device=1, abbreviated=True))
        # model.train() # rnn cannot do backwards in train mode
        # get gradient w.r.t. trigger embeddings for current batch
        data_grad = eval_with_triggers(vm, batch, trigger_token_ids)
        # print(torch.cuda.memory_summary(device=1, abbreviated=True))
        print(np.array(data_grad).shape)
        averaged_grad = np.sum(data_grad, axis=0)
        print(averaged_grad.shape)
        averaged_grad = averaged_grad[1 : len(trigger_token_ids)+1]
        # pass the gradients to a particular attack to generate token candidates for each token.
        

        cand_trigger_token_ids = hotflip_attack_helper(
            averaged_grad,
            embedding_weight,
            num_candidates=40,
            increase_loss=True,
        )
        trigger_token_ids = get_best_candidates(
            vm, batch, trigger_token_ids, cand_trigger_token_ids
        )
        print("after:")
        get_accuracy(vm, batch, trigger_token_ids, True, True)
    get_accuracy(vm, dev_data, trigger_token_ids, True, False)
    return trigger_token_ids
def get_accuracy(
    vm, dev_data, trigger_token_ids, triggers=False, batch=False,
) -> None:
    if batch:
        if triggers:
            print_string = ""
            for idx in trigger_token_ids:
                print_string = (
                    print_string + str(vm.convert_ids_to_tokens(int(idx))) + ", "
                )
            print("triggers:", print_string)
            outputs = eval_with_triggers(vm, dev_data, trigger_token_ids, False)
            logits = outputs[1]
            # print(logits)
            preds = np.argmax(logits, axis=1)
            term = preds == dev_data["labels"].cpu().detach().numpy()
            print("accuracy: ", np.array(term).mean())
        else:
            outputs = vm.get_batch_output(dev_data["input_ids"].cpu().detach().numpy(),dev_data["labels"].cpu().detach().numpy())
            logits = outputs[1]
            # print(logits)
            preds = np.argmax(logits, axis=1)
            term = preds == dev_data["labels"].cpu().detach().numpy()
            print("accuracy: ", np.array(term).mean())
    else:
        train_sampler = RandomSampler(dev_data, replacement=False)
        train_dataloader = DataLoader(
            dev_data,
            batch_size=64,
            sampler=train_sampler,
            collate_fn=default_data_collator,
        )
        if triggers:
            print_string = ""
            for idx in trigger_token_ids:
                print_string = (
                    print_string + str(vm.convert_ids_to_tokens( int(idx))) + ", "
                )
            print("triggers:", print_string)
            with torch.no_grad():
                all_vals = []
                for batch in train_dataloader:
                    outputs = eval_with_triggers(vm, batch, trigger_token_ids, False)
                    logits = outputs[1]
                    preds = np.argmax(logits.cpu().detach().numpy(), axis=1)
                    term = preds == batch["labels"].cpu().detach().numpy()
                    all_vals.extend(term)
                print("accuracy: ", np.array(all_vals).mean())
        else:
            with torch.no_grad():
                all_vals = []
                for batch in train_dataloader:
                    outputs = vm.get_batch_output(batch["input_ids"].cpu().detach().numpy(),batch["labels"].cpu().detach().numpy())
                    logits = outputs[1]
                    preds = np.argmax(logits, axis=1)
                    term = preds == batch["labels"].cpu().detach().numpy()
                    all_vals.extend(term)

            print("accuracy: ", np.array(all_vals).mean())
def eval_with_triggers(
    vm, batch, trigger_token_ids: List[int], gradient=True
) -> Dict[str, Any]:
    """ 
        Evaluate the batch with triggers appended to them. 
        If gradient is true, this function returns the gradient of the input with the appended trigger tokens.
        Can choose whether to return gradient or model output.
    """
    trigger_sequence_tensor = torch.LongTensor(deepcopy(trigger_token_ids))
    # attention_mask_tensor = torch.LongTensor([1, 1, 1])
    # token_type_ids = torch.LongTensor([0, 0, 0])
    # with torch.cuda.device(1):
    trigger_sequence_tensor = trigger_sequence_tensor.repeat(
        len(batch["labels"]), 1
    )
    original_tokens = batch["input_ids"].clone()

    # attention_mask_tensor = attention_mask_tensor.repeat(
        # len(batch["labels"]), 1
    # )
    # original_attention_mask = batch["attention_mask"].clone().to(1)

    # token_type_ids_tensor = token_type_ids.repeat(len(batch["labels"]), 1).to(1)
    # original_token_type_ids = batch["token_type_ids"].clone().to(1)
    # print(original_tokens[:, :1].shape,trigger_sequence_tensor.shape)
    # print(original_tokens[:, 1:].shape)
    original_tokens = torch.cat((original_tokens[:, :1], trigger_sequence_tensor, original_tokens[:, 1:]), 1)
    original_tokens = original_tokens.cpu().detach().numpy()
    # print("evaluate:", original_tokens.shape)
    if gradient:
        data_grad = vm.get_batch_input_gradient(original_tokens, batch["labels"].cpu().detach().numpy())
        return data_grad
    else:
        outputs = vm.get_batch_output(original_tokens, batch["labels"].cpu().detach().numpy())
        return outputs
def get_loss_per_candidate(
    index, vm, batch, trigger_token_ids, cand_trigger_token_ids, snli=False
):
    """
    For a particular index, the function tries all of the candidate tokens for that index.
    The function returns a list containing the candidate triggers it tried, along with their loss.
    """
    if isinstance(cand_trigger_token_ids[0], (np.int64, int)):
        print("Only 1 candidate for index detected, not searching")
        return trigger_token_ids
    loss_per_candidate = []
    # loss for the trigger without trying the candidates
    curr_loss = eval_with_triggers(vm, batch, trigger_token_ids, False)[0]
    loss_per_candidate.append((deepcopy(trigger_token_ids), curr_loss))
    for cand_id in range(len(cand_trigger_token_ids[0])):
        trigger_token_ids_one_replaced = deepcopy(trigger_token_ids)  # copy trigger
        trigger_token_ids_one_replaced[index] = cand_trigger_token_ids[index][
            cand_id
        ]  # replace one token
        loss = eval_with_triggers(vm, batch, trigger_token_ids_one_replaced, False)[0]
        loss_per_candidate.append((deepcopy(trigger_token_ids_one_replaced), loss))
    return loss_per_candidate


def get_best_candidates(
    vm, batch, trigger_token_ids, cand_trigger_token_ids, snli=False, beam_size=1,
) -> List[int]:
    """"
    Given the list of candidate trigger token ids (of number of trigger words by number of candidates
    per word), it finds the best new candidate trigger.
    This performs beam search in a left to right fashion.
    """
    # first round, no beams, just get the loss for each of the candidates in index 0.
    # (indices 1-end are just the old trigger)
    loss_per_candidate = get_loss_per_candidate(
        0, vm, batch, trigger_token_ids, cand_trigger_token_ids, snli
    )
    # maximize the loss
    top_candidates = heapq.nlargest(beam_size, loss_per_candidate, key=itemgetter(1))

    # top_candidates now contains beam_size trigger sequences, each with a different 0th token
    for idx in range(
        1, len(trigger_token_ids)
    ):  # for all trigger tokens, skipping the 0th (we did it above)
        loss_per_candidate = []
        for (
            cand,
            _,
        ) in top_candidates:  # for all the beams, try all the candidates at idx
            loss_per_candidate.extend(
                get_loss_per_candidate(
                    idx, vm, batch, cand, cand_trigger_token_ids, snli
                )
            )
        top_candidates = heapq.nlargest(
            beam_size, loss_per_candidate, key=itemgetter(1)
        )
    return max(top_candidates, key=itemgetter(1))[0]
def hotflip_attack_helper(
    grad, embedding_matrix, increase_loss=False, num_candidates=1,
) -> List[List[int]]:

    grad = torch.FloatTensor(grad)
    embedding_matrix = torch.FloatTensor(embedding_matrix)
    grad = grad.unsqueeze(0)
    print(grad.shape,embedding_matrix.shape)
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
    for idx, instance in enumerate(dev_data):
        # print(instance)
        if instance["label"] == dataset_label_filter:
            targeted_dev_data.append(instance)
    universal_perturb_batch_size = 64
    # tokenizer = model_wrapper.get_tokenizer()
    iterator_dataloader = DataLoader(
        targeted_dev_data,
        batch_size=universal_perturb_batch_size,
        shuffle=True,
        collate_fn=default_data_collator,
    )
    perturbed = universal_trigger_attack(iterator_dataloader,targeted_dev_data,vm,constraint=3)
    print(perturbed)


if __name__ == "__main__":
    main()
