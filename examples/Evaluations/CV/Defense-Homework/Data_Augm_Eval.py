import importlib
import numpy as np
import transformers
from torch.utils.data import DataLoader
from transformers.data.data_collator import default_data_collator

# ------------------ LOCAL IMPORTS ---------------------------------
from Maestro.data import TorchVisionDataset, get_dataset
from Maestro.evaluator import FGSM_Evaluator
from Maestro.models import build_model
from Maestro.attacker_helper.attacker_request_helper import virtual_model
from Maestro.constraints import Epsilon
# ------------------  LOCAL IMPORTS --------------------------------


module = importlib.import_module("Data_Augmentation_Sol")
# url = "http://128.195.56.136:5000"
url = "http://127.0.0.1:5000"
application_name = "Data_Augmentation_CV"
vm = virtual_model(url,application_name=application_name)

# ------------------ SPLIT DATASET --------------------------------
dataset_name = "MNIST"
datasets = get_dataset(dataset_name)
print(datasets)

train_data = datasets["train"]
train_dataset = train_data.get_json_data()
print('train data', len(train_dataset), type(train_dataset))

test_data = datasets["test"]
test_dataset = test_data.get_json_data()
print('test data', len(test_dataset))
targeted_dev_data = test_dataset
targeted_dev_data = targeted_dev_data[:10]
perturb_batch_size = 1
iterator_dataloader = DataLoader(
    targeted_dev_data,
    batch_size=perturb_batch_size,
    shuffle=True,
    collate_fn=default_data_collator,
)
# ------------------ END SPLIT DATASET --------------------------------


# ------------------ DEFENSE TRAINING ---------------------------------
print("Start Training")
response = vm.send_augmented_dataset(train_dataset, module)
print("Augmented dataset received?", response["Done"])
response = vm.send_train_signal()
print("Model trained?", response["Done"])
# ------------------ END DEFENSE TRAINING -----------------------------

# ------------------ ATTACK -------------------------------------------
if response["Done"] == "OK":
    print("Start Evaluation of your Augmented Defense using FGSM attack")
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

    epsilon = 0.2
    constraint = Epsilon(epsilon)
    E = FGSM_Evaluator(module, iterator_dataloader, vm, constraint=constraint)
    E.evaluate_attacker()
else:
    print("There is errors on your data augmentation")
# ------------------ END ATTACK -----------------------------------------
