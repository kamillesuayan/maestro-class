import importlib
import numpy as np
from Maestro.data import HuggingFaceDataset, get_dataset
from Maestro.evaluator import FGSM_Evaluator
from Maestro.models import build_model
from Maestro.attacker_helper.attacker_request_helper import virtual_model
from Maestro.constraints import Epsilon
import transformers
from torch.utils.data import DataLoader
from transformers.data.data_collator import default_data_collator

attacker = importlib.import_module("FGSM_Attack_sol")
# url = "http://128.195.56.136:5000"
url = "http://127.0.0.1:5000"
application_name = "FGSM"
vm = virtual_model(url, application_name=application_name)
vm.ask_result()
"""dataset_name = "MNIST"
datasets = get_dataset(dataset_name)
print(datasets)
test_data = datasets["test"]
test_dataset = test_data.get_json_data()
print(len(test_dataset))
targeted_dev_data = test_dataset
targeted_dev_data = targeted_dev_data[:10]
perturb_batch_size = 1
iterator_dataloader = DataLoader(
    targeted_dev_data,
    batch_size=perturb_batch_size,
    shuffle=True,
    collate_fn=default_data_collator,
)

print("started evaluation")
epsilon = 0.2
constraint = Epsilon(epsilon)
E = FGSM_Evaluator(attacker, iterator_dataloader, vm, constraint=constraint)
E.evaluate_attacker()
"""