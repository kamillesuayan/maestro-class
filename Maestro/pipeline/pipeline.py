import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Iterator, Dict, Tuple, Any, Type
import torch.optim as optim
from functools import wraps
import yaml
from transformers.data.data_collator import default_data_collator

# ------------------ LOCAL IMPORTS ---------------------------------
from Maestro.data import DataModifier
from Maestro.utils import move_to_device, get_embedding
# ------------------ LOCAL IMPORTS ---------------------------------


WRITE_ACCESS = 0


class Scenario:
    """
    Defines the scenario which contains the target and the defense's accesses.
    """

    def __init__(self) -> None:
        self.defense_access = None
        self.target = None
        self.constraint = None

    def load_from_yaml(self, yaml_file) -> None:
        with open(yaml_file) as f:
            data = yaml.load(f)
            self.target = data["Attack Method"]["target"]
            self.constraint = data["Attack Method"]["constraint"]
            self.defense_access = DefenseAccess()
            self.defense_access.load_from_yaml(data["defense Access"])


# def get_access_level(access_dict: Dict[str, bool]) -> int:
#     number = ""
#     for each in access_dict:
#         if access_dict[each]:
#             number += "1"
#         else:
#             number += "0"
#     return int(number, 2)


class DefenseAccess:
    """
    defense defines different access levels.
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
            self.training_data, self.scenario.defense_access.training_data_access_level
        )
        self.validation_data = DataModifier(
            self.validation_data, self.scenario.defense_access.dev_data_access_level
        )
        self.test_data = DataModifier(
            self.test_data, self.scenario.defense_access.test_data_access_level
        )

    def get_batch_output(self, x, data_type="train"):
        assert self.scenario.defense_access.output_access_level["output"] == True
        device = self.device
        x_tensor = torch.FloatTensor(x)
        x_tensor = x_tensor.to(device)
        # print(self.model)
        output = self.model(x_tensor)
        return output

    def get_batch_input_gradient(self, x, data_type="train"):
        assert self.scenario.defense_access.output_access_level["gradient"] == True
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
