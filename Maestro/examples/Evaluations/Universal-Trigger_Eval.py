import importlib
import numpy as np

from Maestro.data import HuggingFaceDataset, get_dataset
from Maestro.evaluator.Trigger_Evaluator import Trigger_Evaluator
from Maestro.models import build_model
from attacker_request_helper import virtual_model

import transformers
from torch.utils.data import DataLoader
from transformers.data.data_collator import default_data_collator

attacker = importlib.import_module("Universal-Trigger_Attack")
# url = "http://128.195.56.136:5000"
url = "http://127.0.0.1:5000"
application_name = "Universal_Attack"
vm = virtual_model(url,application_name)

dataset_name = "SST2"
model_name = "bert-base-uncased"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
test_data = HuggingFaceDataset(
    name="glue", subset="sst2", split="test", label_map=None, shuffle=True
)
print(test_data.examples[0])
test_data.indexed(tokenizer, 128)
test_dataset = test_data.get_json_data()
dataset_label_filter = -1
targeted_dev_data = []
print(len(test_dataset))
for instance in test_dataset:
    if instance["label"] == dataset_label_filter:
        instance["label"] = 0
        targeted_dev_data.append(instance)
print(targeted_dev_data[0])
targeted_dev_data = targeted_dev_data[:10]
universal_perturb_batch_size = 64
iterator_dataloader = DataLoader(
    targeted_dev_data,
    batch_size=universal_perturb_batch_size,
    shuffle=True,
    collate_fn=default_data_collator,
)

print("started evaluation")
E = Trigger_Evaluator(attacker, iterator_dataloader, vm, constraint=3)
E.evaluate_attacker()
