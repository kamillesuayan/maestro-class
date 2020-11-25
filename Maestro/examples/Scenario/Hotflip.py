import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.optim as optim
from copy import deepcopy
from typing import List, Iterator, Dict, Tuple, Any, Type
import numpy as np
import heapq
from operator import itemgetter
from transformers.data.data_collator import default_data_collator
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.data import DataLoader

from Maestro.models import build_model
from Maestro.utils import move_to_device, get_embedding
from Maestro.data import get_dataset
from Maestro.pipeline import (
    AutoPipelineForNLP,
    Pipeline,
    Scenario,
    Attacker,
    model_wrapper,
)
from Maestro.data.HuggingFaceDataset import make_text_dataloader, HuggingFaceDataset
from transformers import (
    Trainer,
    TrainingArguments,
    EvalPrediction,
    glue_compute_metrics,
)

# from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import (
#     StanfordSentimentTreeBankDatasetReader,
# )

# from allennlp.models import Model
# from allennlp.data.vocabulary import Vocabulary
# from allennlp.data.token_indexers import SingleIdTokenIndexer
# from allennlp.training.trainer import Trainer, GradientDescentTrainer
# from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data import PyTorchDataLoader as AllenDataLoader
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.nn.util import get_text_field_mask

# from allennlp.nn.util import move_to_device
from allennlp.common.util import lazy_groups_of

# from torchvision import datasets, transforms
import sys


def process_batch(model_wrapper, batch):
    with torch.no_grad():
        batch = move_to_device(batch, cuda_device=1)
        outputs = model_wrapper.get_batch_output(batch)
    return outputs


def get_accuracy(
    model_wrapper: model_wrapper, dev_data, tokenizer, batch=False,
) -> None:
    # model_wrapper.model.get_metrics(reset=True)
    model_wrapper.model.eval()  # model should be in eval() already, but just in case

    model_wrapper.model.to(1)
    if batch:
        outputs = process_batch(model_wrapper, dev_data)
        logits = outputs[1]
        preds = np.argmax(logits.cpu().detach().numpy(), axis=1)
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
        with torch.no_grad():
            all_vals = []
            for batch in train_dataloader:
                # batch = move_to_device(batch, cuda_device=1)
                outputs = process_batch(model_wrapper, batch)
                logits = outputs[1]
                preds = np.argmax(logits.cpu().detach().numpy(), axis=1)
                term = preds == batch["labels"].cpu().detach().numpy()
                all_vals.extend(term)
            print("accuracy: ", np.array(all_vals).mean())

    model_wrapper.model.train()


def eval_tokens(model_wrapper: nn.Module, batch, gradient=True) -> Dict[str, Any]:
    # if gradient is true, this function returns the gradient of the input with the appended trigger tokens

    def hook_add_trigger_tokens(x):
        return x

    if gradient:
        data_grad = model_wrapper.get_batch_input_gradient(
            batch, hook_add_trigger_tokens
        )
        return data_grad
    else:
        outputs = model_wrapper.get_batch_output(batch, hook_add_trigger_tokens)
        return outputs


def hotflip_attack(
    grad, embedding_matrix, token_ids, increase_loss=False, num_candidates=1,
) -> List[List[int]]:
    """
    The "Hotflip" attack described in Equation (2) of the paper. This code is heavily inspired by
    the nice code of Paul Michel here https://github.com/pmichel31415/translate/blob/paul/
    pytorch_translate/research/adversarial/adversaries/brute_force_adversary.py
    This function takes in the model's average_grad over a batch of examples, the model's
    token embedding matrix, and the current trigger token IDs. It returns the top token
    candidates for each position.
    If increase_loss=True, then the attack reverses the sign of the gradient and tries to increase
    the loss (decrease the model's probability of the true class). For targeted attacks, you want
    to decrease the loss of the target class (increase_loss=False).
    """
    # averaged_grad = averaged_grad.cpu()
    # embedding_matrix = embedding_matrix.cpu()
    grad = move_to_device(torch.from_numpy(grad), 1)
    embedding_matrix = move_to_device(embedding_matrix, 1)
    token_embeds = (
        torch.nn.functional.embedding(
            torch.LongTensor([token_ids]).to(1), embedding_matrix
        )
        .detach()
        .unsqueeze(0)
    )
    grad = grad.unsqueeze(0)
    gradient_dot_embedding_matrix = torch.einsum(
        "bij,kj->bik", (grad, embedding_matrix)
    )
    print(gradient_dot_embedding_matrix.shape)
    if not increase_loss:
        gradient_dot_embedding_matrix *= (
            -1
        )  # lower versus increase the class probability.
    if num_candidates > 1:  # get top k options
        _, best_k_ids = torch.topk(gradient_dot_embedding_matrix, num_candidates, dim=2)
        print(best_k_ids[0])
        print(len(best_k_ids))
        print(len(best_k_ids[0]))
        return best_k_ids.detach().cpu().numpy()[0]
    _, best_at_each_step = gradient_dot_embedding_matrix.max(2)

    return best_at_each_step[0].detach().cpu().numpy()


def get_loss_per_candidate(
    index, model_wrapper, batch, trigger_token_ids, cand_trigger_token_ids, snli=False
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
    curr_loss = (
        eval_with_triggers(model_wrapper, batch, trigger_token_ids, False)[0]
        .cpu()
        .detach()
        .numpy()
    )
    loss_per_candidate.append((deepcopy(trigger_token_ids), curr_loss))
    for cand_id in range(len(cand_trigger_token_ids[0])):
        trigger_token_ids_one_replaced = deepcopy(trigger_token_ids)  # copy trigger
        trigger_token_ids_one_replaced[index] = cand_trigger_token_ids[index][
            cand_id
        ]  # replace one token
        loss = (
            eval_with_triggers(
                model_wrapper, batch, trigger_token_ids_one_replaced, False
            )[0]
            .cpu()
            .detach()
            .numpy()
        )
        loss_per_candidate.append((deepcopy(trigger_token_ids_one_replaced), loss))
    return loss_per_candidate


def get_best_candidates(
    model_wrapper,
    batch,
    trigger_token_ids,
    cand_trigger_token_ids,
    snli=False,
    beam_size=1,
) -> List[int]:
    """"
    Given the list of candidate trigger token ids (of number of trigger words by number of candidates
    per word), it finds the best new candidate trigger.
    This performs beam search in a left to right fashion.
    """
    # first round, no beams, just get the loss for each of the candidates in index 0.
    # (indices 1-end are just the old trigger)
    loss_per_candidate = get_loss_per_candidate(
        0, model_wrapper, batch, trigger_token_ids, cand_trigger_token_ids, snli
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
                    idx, model_wrapper, batch, cand, cand_trigger_token_ids, snli
                )
            )
        top_candidates = heapq.nlargest(
            beam_size, loss_per_candidate, key=itemgetter(1)
        )
    return max(top_candidates, key=itemgetter(1))[0]


def test(model_wrapper, device, num_tokens_change):
    dataset_label_filter = 0
    dev_data = model_wrapper.dev_data.get_write_data()
    targeted_dev_data = []
    for instance in dev_data:
        # print(instance)
        if instance["label"].numpy() == dataset_label_filter:
            targeted_dev_data.append(instance)
    universal_perturb_batch_size = 1
    tokenizer = model_wrapper.get_tokenizer()
    iterator_dataloader = DataLoader(
        targeted_dev_data,
        batch_size=universal_perturb_batch_size,
        shuffle=True,
        collate_fn=default_data_collator,
    )
    print("started the process")
    get_accuracy(model_wrapper, dev_data, tokenizer, batch=False)
    flipped = [0]
    for batch in iterator_dataloader:
        print("start_batch")
        # print(batch)
        get_accuracy(model_wrapper, batch, tokenizer, batch=True)
        # model.train() # rnn cannot do backwards in train mode
        # get gradient w.r.t. trigger embeddings for current batch
        data_grad = eval_tokens(model_wrapper, batch)
        data_grad = data_grad[0][0].detach().cpu().numpy()
        grads_magnitude = [g.dot(g) for g in data_grad]
        print(grads_magnitude)
        # only flip a token once
        for index in flipped:
            grads_magnitude[index] = -1

        # We flip the token with highest gradient norm.
        index_of_token_to_flip = np.argmax(grads_magnitude)
        if grads_magnitude[index_of_token_to_flip] == -1:
            # If we've already flipped all of the tokens, we give up.
            break
        flipped.append(index_of_token_to_flip)
        index_of_token_to_flip = 1

        embedding = get_embedding(model_wrapper.model)
        embedding_weight = embedding.weight.cpu()
        token_ids_to_flip = batch["input_ids"][0][index_of_token_to_flip].to(1)
        grad = data_grad[index_of_token_to_flip]
        print(grad.shape)
        print(token_ids_to_flip)
        cand_trigger_token_ids = hotflip_attack(
            data_grad,
            embedding_weight,
            token_ids_to_flip,
            num_candidates=40,
            increase_loss=True,
        )
        print(cand_trigger_token_ids)
        exit(0)
        trigger_token_ids = get_best_candidates(
            model_wrapper, batch, trigger_token_ids, cand_trigger_token_ids
        )
        print("after:")
        get_accuracy(model_wrapper, batch, tokenizer, True)
    get_accuracy(model_wrapper, targeted_dev_data, tokenizer, False)


def compute_metrics1(p: EvalPrediction) -> Dict:
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": (preds == p.label_ids).mean()}


def compute_metrics(p: EvalPrediction) -> Dict:
    preds = np.argmax(p.predictions, axis=1)
    return glue_compute_metrics("sst-2", preds, p.label_ids)


def main():
    use_cuda = True
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device(
        "cuda:1" if (use_cuda and torch.cuda.is_available()) else "cpu"
    )
    bert = True
    checkpoint_path = ""
    dataset_name = "SST2"
    if bert:
        model_path = "models_temp/" + "BERT_sst2_label/"
        name = "bert-base-uncased"
    else:
        model_path = "models_temp/" + "textattackLSTM/"
        name = "LSTM"
    checkpoint_path = model_path
    training_process = None

    # initialize Atacker, which specifies access rights

    training_data_access = 0
    validation_data_access = 3
    test_data_access = 0
    model_access = 0
    output_access = 2
    myattacker = Attacker(
        training_data_access,
        validation_data_access,
        test_data_access,
        model_access,
        output_access,
    )

    # initialize Scenario. This defines our target
    target = "Instance-Level Adversarial Perturbation"
    myscenario = Scenario(target, myattacker)
    pipeline = AutoPipelineForNLP.initialize(
        name,
        dataset_name,
        model_path,
        checkpoint_path,
        compute_metrics,
        myscenario,
        training_process=None,
        device=1,
        finetune=True,
    )

    model_wrapper = pipeline.get_object()
    test(model_wrapper, device, 5)


if __name__ == "__main__":
    main()
