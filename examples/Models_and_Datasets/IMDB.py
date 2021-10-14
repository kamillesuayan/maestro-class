import dataclasses
import logging
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional
import datasets
import torch
import numpy as np

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    GlueDataset,
)
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)
from transformers.data.data_collator import default_data_collator
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.data import DataLoader
from Maestro.utils import move_to_device, get_embedding

model_name_or_path = "bert-base-uncased"
logging.basicConfig(level=logging.INFO)

task_name = "imdb"
training_args = TrainingArguments(
    output_dir="models_temp/" + task_name,
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    per_gpu_train_batch_size=32,
    per_gpu_eval_batch_size=128,
    num_train_epochs=2,
    evaluation_strategy="epoch",
)
num_labels = 2
config = AutoConfig.from_pretrained(
    model_name_or_path, num_labels=num_labels, finetuning_task=task_name,
)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name_or_path, config=config,
).to(0)
# Get datasets
# train_dataset = GlueDataset(data_args, tokenizer=tokenizer, limit_length=100_000)
# eval_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="dev")
train_dataset = datasets.load_dataset("imdb", split="train")
eval_dataset = datasets.load_dataset("imdb", split="test")
test_dataset = datasets.load_dataset("imdb", split="test")

train_dataset = train_dataset.map(
    lambda e: tokenizer(
        e["text"], max_length=128, truncation=True, padding="max_length",
    ),
    batched=True,
)

train_dataset.set_format(
    type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"]
)
eval_dataset = eval_dataset.map(
    lambda e: tokenizer(
        e["text"], max_length=128, truncation=True, padding="max_length",
    ),
    batched=True,
)
eval_dataset.set_format(
    type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"]
)

test_dataset = test_dataset.map(
    lambda e: tokenizer(
        e["text"], max_length=128, truncation=True, padding="max_length",
    ),
    batched=True,
)
test_dataset.set_format(
    type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"]
)
test_sampler = RandomSampler(test_dataset, replacement=False)
test_dataloader = DataLoader(
    test_dataset, batch_size=64, sampler=test_sampler, collate_fn=default_data_collator,
)


def compute_metrics(p: EvalPrediction) -> Dict:
    preds = np.argmax(p.predictions, axis=1)
    return glue_compute_metrics("sst-2", preds, p.label_ids)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()

with torch.no_grad():
    all_vals = []
    for batch in test_dataloader:
        batch = move_to_device(batch, cuda_device=0)
        outputs = model(**batch)
        logits = outputs[1]
        preds = np.argmax(logits.cpu().detach().numpy(), axis=1)
        term = preds == batch["labels"].cpu().detach().numpy()
        all_vals.extend(term)
    print("accuracy: ", np.array(all_vals).mean())
