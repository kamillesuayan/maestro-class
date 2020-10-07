import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.optim as optim
from copy import deepcopy
from typing import List, Iterator, Dict, Tuple, Any, Type
import numpy
import heapq
from operator import itemgetter

from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import (
    StanfordSentimentTreeBankDatasetReader,
)

from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.training.trainer import Trainer, GradientDescentTrainer
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data import DataLoader
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.nn.util import get_text_field_mask
from allennlp.nn.util import move_to_device
from allennlp.common.util import lazy_groups_of

# from torchvision import datasets, transforms
import sys

sys.path.append("..")
from pipeline import Pipeline, Scenario, Attacker, model_wrapper
from model import build_model
from data import get_data


def get_accuracy(
    model_wrapper: model_wrapper,
    dev_data,
    vocab,
    trigger_token_ids,
    batch=True,
    triggers=False,
) -> None:
    model_wrapper.model.get_metrics(reset=True)
    model_wrapper.model.eval()  # model should be in eval() already, but just in case
    if batch:
        with torch.no_grad():
            batch = move_to_device(dev_data, cuda_device=1)
            model_wrapper.get_batch_output(batch)
    else:
        train_sampler = BucketBatchSampler(
            dev_data, batch_size=128, sorting_keys=["tokens"]
        )
        train_dataloader = DataLoader(dev_data, batch_sampler=train_sampler)
        model_wrapper.model.to(1)
        if triggers:
            print_string = ""
            for idx in trigger_token_ids:
                print_string = print_string + vocab.get_token_from_index(idx) + ", "
            with torch.no_grad():
                for batch in train_dataloader:
                    eval_with_triggers(model_wrapper, batch, trigger_token_ids, False)
        else:
            with torch.no_grad():
                for batch in train_dataloader:
                    batch = move_to_device(batch, cuda_device=1)
                    model_wrapper.get_batch_output(batch)

    print(model_wrapper.model.get_metrics(True)["accuracy"])
    model_wrapper.model.train()


def get_accuracy_with_triggers(
    model_wrapper: model_wrapper, dev_data, vocab, trigger_token_ids
) -> None:
    model_wrapper.model.get_metrics(reset=True)
    model_wrapper.model.eval()  # model should be in eval() already, but just in case
    train_sampler = BucketBatchSampler(
        dev_data, batch_size=128, sorting_keys=["tokens"]
    )
    train_dataloader = DataLoader(dev_data, batch_sampler=train_sampler)
    model_wrapper.model.to(1)
    print_string = ""
    for idx in trigger_token_ids:
        print_string = print_string + vocab.get_token_from_index(idx) + ", "
    with torch.no_grad():
        for batch in train_dataloader:
            eval_with_triggers(model_wrapper, batch, trigger_token_ids, False)

    print(model_wrapper.model.get_metrics(True)["accuracy"])
    model_wrapper.model.train()


def eval_with_triggers(
    model_wrapper: nn.Module, batch, trigger_token_ids: List[int], gradient=True
) -> Dict[str, Any]:
    # if gradient is true, this function returns the gradient of the input with the appended trigger tokens
    trigger_sequence_tensor = torch.LongTensor(deepcopy(trigger_token_ids))
    with torch.cuda.device(1):
        trigger_sequence_tensor = trigger_sequence_tensor.repeat(
            len(batch["label"]), 1
        ).cuda()
        original_tokens = batch["tokens"]["tokens"]["tokens"].clone().cuda()
    batch["tokens"]["tokens"]["tokens"] = torch.cat(
        (trigger_sequence_tensor, original_tokens), 1
    )
    if gradient:
        data_grad = model_wrapper.get_batch_input_gradient(batch)
        batch["tokens"]["tokens"]["tokens"] = original_tokens
        return data_grad
    else:
        outputs = model_wrapper.get_batch_output(batch)
        batch["tokens"]["tokens"]["tokens"] = original_tokens
        return outputs


def hotflip_attack(
    averaged_grad,
    embedding_matrix,
    trigger_token_ids,
    increase_loss=False,
    num_candidates=1,
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
    averaged_grad = averaged_grad.cpu()
    embedding_matrix = embedding_matrix.cpu()
    trigger_token_embeds = (
        torch.nn.functional.embedding(
            torch.LongTensor(trigger_token_ids), embedding_matrix
        )
        .detach()
        .unsqueeze(0)
    )
    averaged_grad = averaged_grad.unsqueeze(0)
    gradient_dot_embedding_matrix = torch.einsum(
        "bij,kj->bik", (averaged_grad, embedding_matrix)
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


def get_loss_per_candidate(
    index, model_wrapper, batch, trigger_token_ids, cand_trigger_token_ids, snli=False
):
    """
    For a particular index, the function tries all of the candidate tokens for that index.
    The function returns a list containing the candidate triggers it tried, along with their loss.
    """
    if isinstance(cand_trigger_token_ids[0], (numpy.int64, int)):
        print("Only 1 candidate for index detected, not searching")
        return trigger_token_ids
    model_wrapper.model.get_metrics(reset=True)
    loss_per_candidate = []
    # loss for the trigger without trying the candidates
    curr_loss = (
        eval_with_triggers(model_wrapper, batch, trigger_token_ids, False)["loss"]
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
            )["loss"]
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


def test(model_wrapper, device, num_tokens_change, vocab):
    dataset_label_filter = "0"
    dev_data = model_wrapper.dev_data.get_write_data()
    print(dev_data)
    targeted_dev_data = []
    for instance, label in dev_data:
        print(instance, label)
        if instance["label"].label == dataset_label_filter:
            targeted_dev_data.append(instance)
    exit(0)
    universal_perturb_batch_size = 128
    num_trigger_tokens = 3
    trigger_token_ids = [vocab.get_token_index("the")] * num_trigger_tokens

    iterator_dataloader = DataLoader(
        targeted_dev_data, batch_size=universal_perturb_batch_size, shuffle=True
    )
    print(iterator_dataloader)
    print("started the process")
    get_accuracy(model_wrapper, dev_data, vocab, trigger_token_ids, False)
    for batch in iterator_dataloader:
        # get accuracy with current triggers
        print("start_batch")
        # print(batch)
        get_accuracy(model_wrapper, batch, vocab, trigger_token_ids, True, True)
        # model.train() # rnn cannot do backwards in train mode

        # get gradient w.r.t. trigger embeddings for current batch
        data_grad = eval_with_triggers(model_wrapper, batch, trigger_token_ids)
        # print(data_grad[0])
        # print(data_grad[0].shape)
        averaged_grad = torch.sum(data_grad[0], dim=0)
        averaged_grad = averaged_grad[0 : len(trigger_token_ids)]
        # print(averaged_grad)
        # print(averaged_grad.shape)
        # pass the gradients to a particular attack to generate token candidates for each token.
        print(model_wrapper.model.word_embeddings._token_embedders)
        embedding_weight = model_wrapper.model.word_embeddings._token_embedders[
            "tokens"
        ].weight.cpu()
        cand_trigger_token_ids = hotflip_attack(
            averaged_grad,
            embedding_weight,
            trigger_token_ids,
            num_candidates=40,
            increase_loss=True,
        )
        print("cand ids", cand_trigger_token_ids)
        # Tries all of the candidates and returns the trigger sequence with highest loss.
        trigger_token_ids = get_best_candidates(
            model_wrapper, batch, trigger_token_ids, cand_trigger_token_ids
        )
    print("triggers: ", trigger_token_ids)
    get_accuracy(model_wrapper, dev_data, vocab, trigger_token_ids, False, True)


def main():
    use_cuda = True
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device(
        "cuda:1" if (use_cuda and torch.cuda.is_available()) else "cpu"
    )
    train_dataloader, validation_dataloader, test_dataloader, vocab = get_data("SST")

    model_path = "models/" + "LSTM/" + "model.th"
    vocab_path = "models/" + "LSTM/" + "vocab"
    if os.path.isfile(model_path):
        name = "LSTM"
        model = build_model(name, model_path, vocab).to(device)
        model.to(1)
    else:
        name = "LSTM"
        model = build_model(name, None, vocab).to(device)
        optimizer = optim.Adam(model.parameters())
        trainer = GradientDescentTrainer(
            model=model,
            optimizer=optimizer,
            data_loader=train_dataloader,
            validation_data_loader=validation_dataloader,
            num_epochs=8,
            patience=1,
            cuda_device=1,
        )
        trainer.train()
        with open(model_path, "wb") as f:
            torch.save(model.state_dict(), f)
        vocab.save_to_files(vocab_path)

    print("CUDA Available: ", torch.cuda.is_available())
    # print("accuracy: ",get_accuracy(model_wrapper,dev_data,vocab))
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    training_process = None

    # initialize Atacker, which specifies access rights

    training_data_access = 0
    dev_data_access = 3
    test_data_access = 0
    model_access = 0
    output_access = 2
    myattacker = Attacker(
        training_data_access,
        dev_data_access,
        test_data_access,
        model_access,
        output_access,
    )

    # initialize Scenario. This defines our target
    target = None
    myscenario = Scenario(target, myattacker)

    model_wrapper = Pipeline(
        myscenario,
        train_dataloader,
        validation_dataloader,
        test_dataloader,
        model,
        training_process,
        device,
    ).get_object()
    test(model_wrapper, device, 5, vocab)


if __name__ == "__main__":
    main()
