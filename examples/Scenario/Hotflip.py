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
# from allennlp.data.samplers import BucketBatchSampler
# from allennlp.data import PyTorchDataLoader as AllenDataLoader
# from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file
# from allennlp.modules.token_embedders import Embedding
# from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
# from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
# from allennlp.nn.util import get_text_field_mask

# from allennlp.nn.util import move_to_device
# from allennlp.common.util import lazy_groups_of

# from torchvision import datasets, transforms
import sys


def process_batch(model_wrapper, batch, data_type):
    with torch.no_grad():
        batch = move_to_device(batch, cuda_device=1)
        outputs = model_wrapper.get_batch_output(batch["uid"], data_type)
    return outputs


def get_accuracy(
    model_wrapper: model_wrapper,
    dev_data,
    tokenizer,
    indexes,
    cand_ids,
    batch=False,
    loader=True,
) -> None:
    # model_wrapper.model.get_metrics(reset=True)
    model_wrapper.model.eval()  # model should be in eval() already, but just in case
    model_wrapper.model.to(1)
    if batch:
        outputs = eval_tokens(
            model_wrapper, dev_data, indexes, cand_ids, "validation", False
        )
        logits = outputs[1]
        preds = np.argmax(logits.cpu().detach().numpy(), axis=1)
        term = preds == dev_data["labels"].cpu().detach().numpy()
        a = np.array(term).mean()
        print("accuracy: ", a)
        return a, logits
    else:
        if loader:
            train_sampler = RandomSampler(dev_data, replacement=False)
            train_dataloader = DataLoader(
                dev_data,
                batch_size=64,
                sampler=train_sampler,
                collate_fn=default_data_collator,
            )
        else:
            train_dataloader = dev_data
        with torch.no_grad():
            all_vals = []
            for batch in train_dataloader:
                # batch = move_to_device(batch, cuda_device=1)
                outputs = eval_tokens(
                    model_wrapper, batch, indexes, cand_ids, "validation", False
                )
                logits = outputs[1]
                preds = np.argmax(logits.cpu().detach().numpy(), axis=1)
                term = preds == batch["labels"].cpu().detach().numpy()
                all_vals.extend(term)
            a = np.array(all_vals).mean()
            print("accuracy: ", a)
            return a

    model_wrapper.model.train()


def eval_tokens(
    model_wrapper: nn.Module, batch, indexes, cand_ids, data_type, gradient=True
) -> Dict[str, Any]:
    # if gradient is true, this function returns the gradient of the input with the appended trigger tokens

    def hook_add_trigger_tokens(x):
        for i, idx in enumerate(indexes):
            x["input_ids"][0][idx] = cand_ids[i]
        return x

    if gradient:
        data_grad = model_wrapper.get_batch_input_gradient(
            batch["uid"], data_type, hook_add_trigger_tokens
        )
        return data_grad
    else:
        outputs = model_wrapper.get_batch_output(
            batch["uid"], data_type, hook_add_trigger_tokens
        )
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
    grad = move_to_device(grad, 1)
    embedding_matrix = move_to_device(embedding_matrix, 1)
    token_embeds = (
        torch.nn.functional.embedding(
            torch.LongTensor([token_ids]).to(1), embedding_matrix
        )
        .detach()
        .unsqueeze(0)
    )
    grad = grad.unsqueeze(0).unsqueeze(0)
    gradient_dot_embedding_matrix = torch.einsum(
        "bij,kj->bik", (grad, embedding_matrix)
    )
    # print(gradient_dot_embedding_matrix.shape)

    # prev_embed_dot_grad = torch.einsum("bij,bij->bi", (grad, token_embeds)).unsqueeze(
    #     -1
    # )
    # # print(prev_embed_dot_grad.shape)
    # neg_dir_dot_grad = 1 * (prev_embed_dot_grad - gradient_dot_embedding_matrix)
    # neg_dir_dot_grad = neg_dir_dot_grad.detach().cpu().numpy()
    # best_at_each_step = neg_dir_dot_grad.argmax(2)
    # return [best_at_each_step[0].data[0]]

    if not increase_loss:
        gradient_dot_embedding_matrix *= (
            -1
        )  # lower versus increase the class probability.
    if num_candidates > 1:  # get top k options
        _, best_k_ids = torch.topk(gradient_dot_embedding_matrix, num_candidates, dim=2)
        return best_k_ids.detach().cpu().numpy()[0]
    _, best_at_each_step = gradient_dot_embedding_matrix.max(2)

    return best_at_each_step[0].detach().cpu().numpy()


def flip_batch(batch, index_of_token_to_flip, cand_ids):
    batch["input_ids"][0][index_of_token_to_flip] = cand_ids[0]
    return batch


def test(model_wrapper, device, num_tokens_change):
    dataset_label_filter = 0
    dev_data = model_wrapper.validation_data.get_write_data()
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
    all_vals = []
    indexes = []
    cand_ids = []
    og_acc = get_accuracy(
        model_wrapper, dev_data, tokenizer, indexes, cand_ids, batch=False
    )
    new_batches = []
    for batch in iterator_dataloader:
        print("start_batch")
        # print(batch)
        flipped = [0]
        acc, oglogits = get_accuracy(
            model_wrapper, batch, tokenizer, indexes, cand_ids, batch=True
        )
        indexes = []
        cand_ids = []
        for i in range(1):
            # model.train() # rnn cannot do backwards in train mode
            # get gradient w.r.t. trigger embeddings for current batch
            data_grad = eval_tokens(
                model_wrapper, batch, indexes, cand_ids, "validation", True
            )
            data_grad = data_grad[0]
            grads_magnitude = [g.dot(g) for g in data_grad]

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
            embedding = get_embedding(model_wrapper.model)
            embedding_weight = embedding.weight.cpu()
            token_ids_to_flip = batch["input_ids"][0][index_of_token_to_flip].to(1)
            grad = data_grad[index_of_token_to_flip]
            indexes.append(index_of_token_to_flip)
            cand_id = hotflip_attack(
                grad,
                embedding_weight,
                token_ids_to_flip,
                num_candidates=1,
                increase_loss=True,
            )
            print("cand ids:", cand_id)
            cand_ids.append(cand_id[0])
            acc, logits = eval_tokens(
                model_wrapper, batch, indexes, cand_ids, "validation", False
            )
            logits = logits.cpu().detach().numpy()
            print("og label", batch["labels"].cpu().detach().numpy()[0])
            print("oglogits: {}, logits: {}".format(oglogits[0], logits[0]))
            print()
        preds = np.argmax(logits, axis=1)
        term = preds == batch["labels"].cpu().detach().numpy()
        all_vals.extend(term)
        new_batches.append(batch)
    print("After attack accuracy of validation dataset:")
    a = np.array(all_vals).mean()
    print("accuracy: ", a)
    # get_accuracy(model_wrapper, new_batches, tokenizer, indexes, cand_ids, False, False)
    print("og accuracy =", og_acc)


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
