import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Iterator, Dict, Tuple, Any, Type
# from torchvision import datasets, transforms
from functools import wraps

# not supposed to use this
from allennlp.nn.util import move_to_device
class Scenario:
    def __init__(self, target,attacker):
        self.target = target
        self.attacker = attacker
class Attacker:
    def __init__(self, training_data_access: bool,test_data_access:bool,model_access_level:int,output_access:bool):
        self.training_data_access = training_data_access
        self.test_data_access = test_data_access
        self.model_access_level = model_access_level
        self.output_access = output_access 

class model_wrapper:
    def __init__(self, model:nn.Module, scenario: Scenario,training_data:, test_data, training_process,device):
        self.model = model
        self.scenario = scenario
        self.device = device

def add_method(cls):
    def decorator(func):
        @wraps(func) 
        def wrapper(self, *args, **kwargs): 
            return func(*args, **kwargs)
        setattr(cls, func.__name__, wrapper)
        return func # returning func means func can still be used normally
    return decorator
class Pipeline:
    # This class is the most abstact class
    def __init__(self,scenario: Scenario,training_data, test_data, model: nn.Module, training_process,device: int):
        self.scenario = scenario
        self.training_data = training_data
        self.test_data = test_data
        self.model = model
        self.training_process = training_process
        self.device = device
    def get_object(self) -> model_wrapper:
        obj = model_wrapper(self.model, self.scenario,self.training_data, self.test_data, self.training_process,self.device)
        if self.scenario.attacker.output_access == True:
            @add_method(model_wrapper)
            def get_input_gradient(x):
                device = self.device
                x= x.to(device)
                x.requires_grad = True
                output = self.model(x)
                pred = output.max(1, keepdim=True)[1]
                loss = F.nll_loss(output, pred[0])
                self.model.zero_grad()
                loss.backward()
                x_grad = x.grad.data
                return x_grad
            @add_method(model_wrapper)
            def get_batch_input_gradient(x):
                device = self.device
                x = move_to_device(x, cuda_device=1)
                embedding_outputs = []
                def hook_layers(module, grad_in, grad_out):
                    # print("just pass",grad_out.shape)
                    grad_out.requires_grad = True
                    embedding_outputs.append(grad_out)
                hooks = []
                hooks.append(self.model.word_embeddings.register_forward_hook(hook_layers))
                outputs = self.model.forward(x['tokens'], x['label'])
                loss = outputs["loss"]
                embedding_gradients_auto = torch.autograd.grad(loss, embedding_outputs[0],create_graph=False)
                # print(x["tokens"]["tokens"]["tokens"])
                # print(embedding_gradients_auto)
                return embedding_gradients_auto
            @add_method(model_wrapper)
            def get_output(x):
                device = self.device
                x = x.to(device)
                output = self.model(x)
                return output
            @add_method(model_wrapper)
            def get_batch_output(x):
                device = self.device
                batch = move_to_device(x, cuda_device=1)
                output = self.model(batch['tokens'], batch['label'])
                return output
        return obj

