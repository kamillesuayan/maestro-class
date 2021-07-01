from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union
import torch
import json


def get_json_data(examples):
    new_data = []
    for idx, instance in enumerate(examples):
        new_instance = {}
        new_instance["image"] = instance[0].numpy().tolist()
        new_instance["label"] = instance[1]
        new_instance["uid"] = idx
        new_data.append(new_instance)
    return new_data


def int_to_device(device: Union[int, torch.device]) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device < 0:
        return torch.device("cpu")
    return torch.device(device)


def has_tensor(obj) -> bool:
    """
    Given a possibly complex data structure,
    check if it has any torch.Tensors in it.
    """
    if isinstance(obj, torch.Tensor):
        return True
    elif isinstance(obj, dict):
        return any(has_tensor(value) for value in obj.values())
    elif isinstance(obj, (list, tuple)):
        return any(has_tensor(item) for item in obj)
    else:
        return False


def move_to_device(obj, cuda_device: Union[torch.device, int]):
    """
    Referenced from Allennlp
    Given a structure (possibly) containing Tensors on the CPU,
    move all the Tensors to the specified GPU (or do nothing, if they should be on the CPU).
    """

    cuda_device = int_to_device(cuda_device)

    if cuda_device == torch.device("cpu") or not has_tensor(obj):
        return obj
    elif isinstance(obj, torch.Tensor):
        return obj.cuda(cuda_device)
    elif isinstance(obj, dict):
        return {key: move_to_device(value, cuda_device) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(item, cuda_device) for item in obj]
    elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
        # This is the best way to detect a NamedTuple, it turns out.
        return obj.__class__(*(move_to_device(item, cuda_device) for item in obj))
    elif isinstance(obj, tuple):
        return tuple(move_to_device(item, cuda_device) for item in obj)
    else:
        return obj


def list_to_json(x):
    """ assumes x to be a list of objects"""
    for idx, element in enumerate(x):
        if isinstance(element, torch.Tensor):
            x[idx] = element.detach().cpu().numpy().tolist()
        elif isinstance(element, list):
            # x[idx] = list_to_json(element)
            x[idx] = element

    returned = json.dumps(x)
    # print("list to json:", type(x))
    # print("list to json:", type(returned))
    return returned


def get_embedding(model):
    embedding = None
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            embedding = module
            break
    return embedding
