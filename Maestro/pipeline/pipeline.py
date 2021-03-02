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

    def __init__(self, target, attacker) -> None:
        self.target = target
        self.attacker = attacker


def get_access_level(access_dict: Dict[str, bool]) -> int:
    number = ""
    for each in access_dict:
        if access_dict[each]:
            number += "1"
        else:
            number += "0"
    return int(number, 2)


class Attacker:
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

    def load_from_yaml(self, yaml_file) -> None:
        with open(yaml_file) as f:
            data = yaml.load(f)
            print(data)
            self.training_data_access_level = get_access_level(
                data["Attacker"]["training_data_access"]
            )
            self.dev_data_access_level = get_access_level(
                data["Attacker"]["dev_data_access"]
            )
            self.test_data_access_level = get_access_level(
                data["Attacker"]["test_data_access"]
            )
            self.model_access_level = get_access_level(data["Attacker"]["model_access"])
            self.output_access_level = get_access_level(
                data["Attacker"]["output_access"]
            )


class model_wrapper:
    """
    model_wraper is an object that only contains method/data that are allowed to the users.
    """

    def __init__(
        self,
        model: nn.Module,
        scenario: Scenario,
        train_data,
        dev_data,
        test_data,
        training_process,
        device,
    ) -> None:
        self.model = model
        self.scenario = scenario
        self.device = device
        # self.train_data = train_data
        # self.dev_data = dev_data
        # self.test_data = test_data


def add_method(cls):
    """
    A function that adds method to an object.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(*args, **kwargs)

        setattr(cls, func.__name__, wrapper)
        return func  # returning func means func can still be used normally

    return decorator


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

    def get_object(self) -> model_wrapper:
        obj = model_wrapper(
            self.model,
            self.scenario,
            self.training_data,
            self.validation_data,
            self.test_data,
            self.training_process,
            self.device,
        )
        # adding methods for getting the prediction and the outputs
        if self.scenario.attacker.output_access_level > 0:

            @add_method(model_wrapper)
            def _get_inputs(x_idx, data_type="train", nlp=True):
                data = None
                if data_type == "train":
                    data = self.training_data[x_idx]
                elif data_type == "validation":
                    data = self.validation_data[x_idx]
                elif data_type == "test":
                    data = self.test_data[x_idx]
                if nlp:
                    data = default_data_collator(data)
                    # print("get_inputs:", data)
                    # print("training:", self.training_data[x_idx])
                    # print("get_inputs:", data["uid"])
                    del data["uid"]
                else:
                    data = data[0]
                return data

            @add_method(model_wrapper)
            def get_output(x_id, data_type="train", pred_hook=lambda x: x):
                device = self.device
                x = obj._get_inputs(x_id, data_type, nlp=False)
                x = x.unsqueeze(0)
                x = x.to(device)
                print(pred_hook(x))
                output = self.model(pred_hook(x))
                return output

            @add_method(model_wrapper)
            def get_batch_output(
                x_ids, data_type="train", pred_hook=lambda x: x
            ) -> Dict[str, Any]:
                # TODO Combine this with the top ones. This may get complicated as this method needs to handle for both NLP and CV
                device = self.device
                x = obj._get_inputs(x_ids, data_type)
                with torch.no_grad():
                    batch = move_to_device(pred_hook(x), cuda_device=device)
                    # print("get batch output:", batch)
                    output = self.model(**batch)
                return output

            if self.scenario.attacker.output_access_level > 1:

                @add_method(model_wrapper)
                def get_input_gradient(x_id, data_type="train", pred_hook=lambda x: x):
                    device = self.device
                    x = obj._get_inputs(x_id, data_type, nlp=False)
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

                @add_method(model_wrapper)
                def get_batch_input_gradient(
                    x_ids, data_type="train", pred_hook=lambda x: x
                ):
                    device = self.device
                    embedding_outputs = []

                    def hook_layers(module, grad_in, grad_out):
                        # grad_out.requires_grad = True
                        embedding_outputs.append(grad_out)

                    hooks = []
                    embedding = get_embedding(self.model)
                    hooks.append(embedding.register_forward_hook(hook_layers))
                    # print(torch.cuda.memory_summary(device=0, abbreviated=True))

                    x = obj._get_inputs(x_ids, data_type)
                    # print(x["input_ids"])
                    # print(pred_hook(x)["input_ids"])
                    outputs = self.model(
                        **move_to_device(pred_hook(x), cuda_device=device)
                    )
                    loss = outputs[0]
                    embedding_gradients_auto = torch.autograd.grad(
                        loss, embedding_outputs[0], create_graph=False
                    )
                    for hook in hooks:
                        hook.remove()
                    return embedding_gradients_auto[0]

        # getting the data modifier
        obj.training_data = DataModifier(
            self.training_data, self.scenario.attacker.training_data_access_level
        )
        obj.validation_data = DataModifier(
            self.validation_data, self.scenario.attacker.dev_data_access_level
        )
        obj.test_data = DataModifier(
            self.test_data, self.scenario.attacker.test_data_access_level
        )

        @add_method(model_wrapper)
        def get_tokenizer():
            return self.tokenizer

        # adding the training method
        # or we could return a trainer object
        @add_method(model_wrapper)
        def train():
            optimizer = optim.Adam(self.model.parameters())
            train_sampler = BucketBatchSampler(
                self.training_data, batch_size=32, sorting_keys=["tokens"]
            )
            validation_sampler = BucketBatchSampler(
                self.dev_data, batch_size=32, sorting_keys=["tokens"]
            )
            train_dataloader = DataLoader(
                self.training_data, batch_sampler=train_sampler
            )
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

        return obj
