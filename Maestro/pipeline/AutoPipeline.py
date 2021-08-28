import os
from Maestro.pipeline import Pipeline, VisionPipeline
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

import torch.nn as nn
import torch.optim as optim



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
        model = build_model(model_name, num_labels=2, max_length=128, device=device)
        self.tokenizer = model.tokenizer
        self.device = device
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

        # test_sampler = RandomSampler(validation_dataset, replacement=False)
        # test_dataloader = DataLoader(
        #     validation_dataset,
        #     sampler=test_sampler,
        #     batch_size=32,
        #     collate_fn=default_data_collator,
        # )
        # model.to(self.device)
        # print(torch.cuda.memory_summary(device=1, abbreviated=True))
        # all_vals = []
        # with torch.no_grad():
        #     for batch in test_dataloader:
        #         batch = move_to_device(batch, cuda_device=self.device)
        #         del batch["uid"]
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
        model = build_model(model_name, num_labels=2, max_length=128, device=device)
        self.device = device
        train_dataset = datasets["train"]
        test_dataset = datasets["test"]
        if finetune:
            model = AutoPipelineForVision.fine_tune_on_task(
                AutoPipelineForVision,
                model,
                train_dataset,
                test_dataset,
                model_path,
                checkpoint_path,
                compute_metrics=compute_metrics,
            )
        return VisionPipeline(
            scenario,
            train_dataset,
            test_dataset,
            test_dataset,
            model,
            training_process,
            device,
            None,
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
        print(os.listdir(checkpoint_path))
        if not model_path or len(os.listdir(checkpoint_path)) == 0:
            print("start training")
            pass
        else:
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        return model


class AutoPipelineForSec:
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
        finetune=True):
        print(dataset_name)

        datasets = get_dataset(dataset_name)
        self.model = build_model(model_name, num_labels=2, max_length=128, device=0)
        self.model.to(device)
        train_dataset = datasets["train"]
        test_dataset = datasets["test"]

        if not model_path or len(os.listdir(checkpoint_path)) == 0:
            self.model = self.train(self.model, train_dataset, device)
            torch.save(self.model.state_dict(), checkpoint_path+'malimg.pth')

        else:
            self.model.load_state_dict(torch.load(model_path, map_location="cuda:"+str(device)))
            print("train:")
            self.test(self.model, train_dataset, device)
            print("test:")
            self.test(self.model, test_dataset, device)

        tokenizer = 0
        # print(train_dataset)
        # print(test_dataset)
        return VisionPipeline(
            scenario,
            train_dataset,
            test_dataset,
            test_dataset,
            self.model,
            training_process,
            device,
            tokenizer)



    @classmethod
    def train(self, model, trainset, device, epoches=10):
        model.train()
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=10)
        dataset_size = len(trainset)
        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        optimizer = optim.Adam(model.parameters())
        for epoch in range(epoches):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
            print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / dataset_size))
            running_loss = 0.0
        self.test(model, trainset, device)

        return model

    def test(model, testset, device):
        model.eval()
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True, num_workers=10)
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the images: %.3f %%' % (
            100*correct / total))
        return
