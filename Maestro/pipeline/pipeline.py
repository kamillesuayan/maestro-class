import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Iterator, Dict, Tuple, Any, Type
import torch.optim as optim
from functools import wraps
import yaml
from Maestro.data import DataModifier
from Maestro.utils import move_to_device, get_embedding
from transformers.data.data_collator import default_data_collator

# not supposed to use this
# from allennlp.nn.util import move_to_device
# from allennlp.data.samplers import BucketBatchSampler
# from allennlp.data import DataLoader
# from allennlp.training.trainer import Trainer, GradientDescentTrainer

WRITE_ACCESS = 0


class Scenario:
    """
    Defines the scenario which contains the target and the attacker's accesses.
    """

    def __init__(self) -> None:
        self.attacker_access = None
        self.target = None
        self.constraint = None

    def load_from_yaml(self, yaml_file) -> None:
        with open(yaml_file) as f:
            data = yaml.load(f)
            self.target = data["Attack Method"]["target"]
            self.constraint = data["Attack Method"]["constraint"]
            self.attacker_access = AttackerAccess()
            self.attacker_access.load_from_yaml(data["Attacker Access"])


# def get_access_level(access_dict: Dict[str, bool]) -> int:
#     number = ""
#     for each in access_dict:
#         if access_dict[each]:
#             number += "1"
#         else:
#             number += "0"
#     return int(number, 2)


class AttackerAccess:
    """
    Attacker defines different access levels.
    """

    def __init__(
        self,
        training_data_access_level: int = None,
        dev_data_access_level: int = None,
        test_data_access_level: int = None,
        model_access_level: int = None,
        output_access_level: int = None,
    ) -> None:
        # 0 = no acess,1 = write acess,2 = read acess and 3 = write/read access
        self.training_data_access_level = training_data_access_level
        self.dev_data_access_level = dev_data_access_level
        self.test_data_access_level = test_data_access_level
        self.model_access_level = model_access_level

        # 0 = no acess,1 = output access only, 2 = gradient access only, 2 = output acess and gradient access
        self.output_access_level = output_access_level

    def load_from_yaml(self, data) -> None:

        self.training_data_access_level = data["training_data_access"]
        self.dev_data_access_level = data["dev_data_access"]
        self.test_data_access_level = data["test_data_access"]
        self.model_access_level = data["model_access"]
        self.output_access_level = data["output_access"]


class Pipeline:
    """
    Pipeline contains everything.
    """

    def __init__(
        self,
        scenario: Scenario,
        training_data,
        validation_data,
        test_data,
        model: nn.Module,
        training_process,
        device: int,
        tokenizer,
    ) -> None:
        self.scenario = scenario
        self.training_data = training_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.model = model
        self.training_process = training_process
        self.device = device
        self.tokenizer = tokenizer

        # adding methods for getting the prediction and the outputs
        # getting the data modifier
        self.training_data = DataModifier(
            self.training_data, self.scenario.attacker_access.training_data_access_level
        )
        self.validation_data = DataModifier(
            self.validation_data, self.scenario.attacker_access.dev_data_access_level
        )
        self.test_data = DataModifier(
            self.test_data, self.scenario.attacker_access.test_data_access_level
        )

    def _get_inputs(self, x, data_type="train", nlp=True):
        data = None
        if nlp:
            data = default_data_collator(x)
            # print("get_inputs:", data)
            # print("training:", self.training_data[x_idx])
            # print("get_inputs:", data["uid"])
            del data["uid"]
        else:
            data = x[0]
        return data

    def get_output(self, x_id, data_type="train", pred_hook=lambda x: x):
        assert self.scenario.attacker_access.output_access_level["output"] == True
        device = self.device
        x = self._get_inputs(x_id, data_type, nlp=False)
        x = x.unsqueeze(0)
        x = x.to(device)
        # print(pred_hook(x))
        output = self.model(pred_hook(x))
        return output

    def get_batch_output(self, x, labels) -> Dict[str, Any]:
        # TODO Combine this with the top ones. This may get complicated as this method needs to handle for both NLP and CV
        assert self.scenario.attacker_access.output_access_level["output"] == True
        device = self.device
        # print(x)
        decoded_x = self.tokenizer.batch_decode(x, skip_special_tokens=True)
        x = self.tokenizer.batch_encode_plus(
            decoded_x,
            max_length=128,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)
        # x = obj._get_inputs(x, data_type)
        x["labels"] = torch.LongTensor(labels).to(device)
        with torch.no_grad():
            print("get batch output:", x["input_ids"].shape)
            output = self.model(**x)
        return output

    def get_input_gradient(self, x_id, data_type="train", pred_hook=lambda x: x):
        assert self.scenario.attacker_access.output_access_level["gradient"] == True
        device = self.device
        x = self._get_inputs(x_id, data_type, nlp=False)
        x = x.unsqueeze(0)
        x = x.to(device)
        x.requires_grad = True
        transformed_x = pred_hook(x)
        output = self.model(transformed_x)
        pred = output.max(1, keepdim=True)[1]
        loss = F.nll_loss(output, pred[0])
        self.model.zero_grad()
        loss.backward()
        x_grad = x.grad.data
        print("pipeline")
        print(x_grad)
        return x_grad

    def get_batch_input_gradient(self, x, labels):
        assert self.scenario.attacker_access.output_access_level["gradient"] == True
        device = self.device
        embedding_outputs = []

        def hook_layers(module, grad_in, grad_out):
            # grad_out.requires_grad = True
            embedding_outputs.append(grad_out)

        hooks = []
        embedding = get_embedding(self.model)
        hooks.append(embedding.register_forward_hook(hook_layers))
        # print(torch.cuda.memory_summary(device=0, abbreviated=True))

        decoded_x = self.tokenizer.batch_decode(x, skip_special_tokens=True)
        print(decoded_x)
        x = self.tokenizer.batch_encode_plus(
            decoded_x,
            max_length=128,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)
        # print(x["input_ids"])
        # print(pred_hook(x)["input_ids"])
        x["labels"] = torch.LongTensor(labels).to(device)
        outputs = self.model(**x)
        # print(outputs)
        loss = outputs[0]
        # print(loss)
        embedding_gradients_auto = torch.autograd.grad(
            loss, embedding_outputs[0], create_graph=False
        )
        for hook in hooks:
            hook.remove()
        return embedding_gradients_auto[0]

    def get_tokenizer(self):
        return self.tokenizer

    # adding the training method
    # or we could return a trainer object
    def train(self):
        optimizer = optim.Adam(self.model.parameters())
        train_sampler = BucketBatchSampler(
            self.training_data, batch_size=32, sorting_keys=["tokens"]
        )
        validation_sampler = BucketBatchSampler(
            self.dev_data, batch_size=32, sorting_keys=["tokens"]
        )
        train_dataloader = DataLoader(self.training_data, batch_sampler=train_sampler)
        validation_dataloader = DataLoader(
            self.dev_data, batch_sampler=validation_sampler
        )
        trainer = GradientDescentTrainer(
            model=self.model,
            optimizer=optimizer,
            data_loader=train_dataloader,
            validation_data_loader=validation_dataloader,
            num_epochs=8,
            patience=1,
            cuda_device=1,
        )
        trainer.train()
        return self.test_data


class VisionPipeline:
    """
    Pipeline contains everything.
    """

    def __init__(
        self,
        scenario: Scenario,
        training_data,
        validation_data,
        test_data,
        model: nn.Module,
        training_process,
        device: int,
        tokenizer,
    ) -> None:
        self.scenario = scenario
        self.training_data = training_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.model = model
        self.training_process = training_process
        self.device = device
        self.tokenizer = tokenizer

        # adding methods for getting the prediction and the outputs
        # getting the data modifier
        self.training_data = DataModifier(
            self.training_data, self.scenario.attacker_access.training_data_access_level
        )
        self.validation_data = DataModifier(
            self.validation_data, self.scenario.attacker_access.dev_data_access_level
        )
        self.test_data = DataModifier(
            self.test_data, self.scenario.attacker_access.test_data_access_level
        )

    def get_batch_output(self, x, data_type="train"):
        assert self.scenario.attacker_access.output_access_level["output"] == True
        device = self.device
        x_tensor = torch.FloatTensor(x)
        x_tensor = x_tensor.to(device)
        # print(self.model)
        output = self.model(x_tensor)
        return output

    def get_batch_input_gradient(self, x, data_type="train"):
        assert self.scenario.attacker_access.output_access_level["gradient"] == True
        device = self.device
        x_tensor = torch.FloatTensor(x)
        x_tensor = x_tensor.to(device)
        x_tensor.requires_grad = True
        print(x_tensor.shape)
        output = self.model(x_tensor)
        pred = output.max(1, keepdim=True)[1]
        loss = F.nll_loss(output, pred[0])
        self.model.zero_grad()
        loss.backward()
        x_grad = x_tensor.grad.data
        print("pipeline, get_batch_input_gradient")
        # print(x_grad)
        return x_grad

