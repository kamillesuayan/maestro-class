import os
from Maestro.pipeline import Pipeline
from Maestro.data import HuggingFaceDataset, get_dataset
from Maestro.models import build_model
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.data import DataLoader
from transformers import (
    Trainer,
    TrainingArguments,
    EvalPrediction,
    glue_compute_metrics,
)
from transformers.data.data_collator import default_data_collator
from Maestro.utils import move_to_device, get_embedding
import numpy as np
import torch


class AutoPipelineForNLP:
    def __init__(self):
        raise EnvironmentError("Use this like the AutoModel from huggingface")

    @classmethod
    def initialize(
        self,
        model_name,
        dataset_name,
        model_path,
        checkpoint_path,
        compute_metrics,
        scenario,
        training_process=None,
        device=0,
        finetune=True,
    ):
        datasets = get_dataset(dataset_name)
        model = build_model(model_name, num_labels=2, max_length=128, device=1)
        self.tokenizer = model.tokenizer
        for dataset in datasets:
            dataset.indexed(self.tokenizer, 128)

        train_dataset = datasets[0]
        if len(datasets) == 2:
            validation_dataset = datasets[1]
            test_dataset = datasets[1]
        else:
            validation_dataset = datasets[1]
            test_dataset = datasets[2]
        if finetune:
            model = AutoPipelineForNLP.fine_tune_on_task(
                AutoPipelineForNLP,
                model,
                train_dataset,
                validation_dataset,
                model_path,
                checkpoint_path,
                compute_metrics=compute_metrics,
            )
        return Pipeline(
            scenario,
            train_dataset,
            validation_dataset,
            validation_dataset,
            model,
            training_process,
            device,
            self.tokenizer,
        )

    def fine_tune_on_task(
        self,
        model,
        train_dataset,
        validation_dataset,
        model_path,
        checkpoint_path,
        compute_metrics=None,
    ):
        if not model_path or len(os.listdir(model_path)) == 0:
            print("start training")
            # optimizer = optim.Adam(model.model.parameters())
            # scheduler = optim.lr_scheduler.LambdaLR(optimizer)
            training_args = TrainingArguments(
                output_dir=checkpoint_path,
                do_train=True,
                do_eval=True,
                num_train_epochs=1,
                evaluation_strategy="steps",
                save_steps=200,
                eval_steps=200,
                per_device_train_batch_size=32,
                per_device_eval_batch_size=128,
                save_total_limit=5,
            )
            trainer = Trainer(
                args=training_args,
                model=model.model,
                # optimizers=(optimizer, None),
                train_dataset=train_dataset.get_trainable_data(self.tokenizer, 128),
                eval_dataset=validation_dataset.get_trainable_data(self.tokenizer, 128),
                compute_metrics=compute_metrics,
            )
            trainer.train()
            trainer.save_model(checkpoint_path)
            model = model.model
        else:
            print("loading model from", model_path)
            model = model.model.from_pretrained(model_path)

        test_sampler = RandomSampler(validation_dataset, replacement=False)
        test_dataloader = DataLoader(
            validation_dataset,
            sampler=test_sampler,
            batch_size=32,
            collate_fn=default_data_collator,
        )
        model.to(1)
        # print(torch.cuda.memory_summary(device=1, abbreviated=True))
        # all_vals = []
        # with torch.no_grad():
        #     for batch in test_dataloader:
        #         batch = move_to_device(batch, cuda_device=1)
        #         outputs = model(**batch)
        #         logits = outputs[1]
        #         preds = np.argmax(logits.cpu().detach().numpy(), axis=1)
        #         term = preds == batch["labels"].cpu().detach().numpy()
        #         all_vals.extend(term)
        # print("accuracy: ", np.array(all_vals).mean())
        # print(torch.cuda.memory_summary(device=1, abbreviated=True))

        return model


class AutoPipelineForVision:
    def __init__(self):
        raise EnvironmentError("Use this like the AutoModel from huggingface")

    @classmethod
    def initialize(
        self,
        model_name,
        dataset_name,
        model_path,
        checkpoint_path,
        compute_metrics,
        scenario,
        training_process=None,
        device=0,
        finetune=True,
    ):
        datasets = get_dataset(dataset_name)
        model = build_model(model_name, num_labels=2, max_length=128, device=1)
        tokenizer = model.tokenizer
        for dataset in datasets:
            dataset_huggingface = dataset.indexed(tokenizer, 128)

        train_dataset = datasets[0]
        if len(datasets) == 2:
            validation_dataset = datasets[1]
            test_dataset = datasets[1]
        else:
            validation_dataset = datasets[1]
            test_dataset = datasets[2]
        if finetune:
            model = AutoPipeline.fine_tune_on_task(
                AutoPipeline,
                model,
                train_dataset,
                validation_dataset,
                model_path,
                checkpoint_path,
                compute_metrics=compute_metrics,
            )
        return Pipeline(
            scenario,
            train_dataset,
            validation_dataset,
            validation_dataset,
            model,
            training_process,
            device,
            tokenizer,
        )

    def fine_tune_on_task(
        self,
        model,
        train_dataset,
        validation_dataset,
        model_path,
        checkpoint_path,
        compute_metrics=None,
    ):
        if not model_path:
            print("start training")
            # optimizer = optim.Adam(model.model.parameters())
            # scheduler = optim.lr_scheduler.LambdaLR(optimizer)
            training_args = TrainingArguments(
                output_dir=checkpoint_path,
                do_train=True,
                do_eval=True,
                num_train_epochs=1,
                evaluation_strategy="steps",
                save_steps=200,
                eval_steps=200,
                per_device_train_batch_size=32,
                per_device_eval_batch_size=128,
                save_total_limit=5,
            )
            trainer = Trainer(
                args=training_args,
                model=model.model,
                # optimizers=(optimizer, None),
                train_dataset=train_dataset,
                eval_dataset=validation_dataset,
                compute_metrics=compute_metrics,
            )
            trainer.train()
            trainer.save_model(checkpoint_path)
            model = model.model
        else:
            print("loading model from", model_path)
            model = model.model.from_pretrained(model_path)
        test_sampler = RandomSampler(validation_dataset, replacement=False)
        test_dataloader = DataLoader(
            validation_dataset,
            sampler=test_sampler,
            batch_size=32,
            collate_fn=default_data_collator,
        )
        model.to(1)
        # all_vals = []
        # for batch in test_dataloader:
        #     batch = move_to_device(batch, cuda_device=1)
        #     outputs = model(**batch)
        #     logits = outputs[1]
        #     preds = np.argmax(logits.cpu().detach().numpy(), axis=1)
        #     term = preds == batch["labels"].cpu().detach().numpy()
        #     all_vals.extend(term)
        # print("accuracy: ", np.array(all_vals).mean())

        return model
