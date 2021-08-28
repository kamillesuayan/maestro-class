import importlib
import numpy as np
import transformers
from torch.utils.data import DataLoader
from transformers.data.data_collator import default_data_collator

# ------------------ LOCAL IMPORTS ---------------------------------
from Maestro.data import HuggingFaceDataset, get_dataset
from Maestro.evaluator import FGSM_Evaluator
from Maestro.models import build_model
from Maestro.defense_helper.defense_request_helper import virtual_model
from Maestro.constraints import Epsilon
from Maestro.trainers import Data_Augmentation_Trainer
# ------------------  LOCAL IMPORTS --------------------------------


module = importlib.import_module("Data_Augmentation_Sol")
# url = "http://128.195.56.136:5000"
url = "http://127.0.0.1:5000"
application_name = "FGSM"
vm = virtual_model(url,application_name=application_name)

# ------------------ SPLIT DATASET --------------------------------
dataset_name = "MNIST"
datasets = get_dataset(dataset_name)
print(datasets)

train_data = datasets["train"]
train_dataset = train_data.get_json_data()
print('train data', len(train_dataset))
targeted_train_data = train_dataset
#### RESEARCH ABOUT THIS BATCH SIZE!!!
perturb_batch_size = 100
train_dataloader = DataLoader(
    targeted_train_data,
    batch_size=perturb_batch_size,
    shuffle=True,
    collate_fn=default_data_collator,
)


test_data = datasets["test"]
test_dataset = test_data.get_json_data()
print('train data', len(test_dataset))
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
T = Data_Augmentation_Trainer(module, train_dataloader, vm, df_constraint=df_constraint)
response = T.train()
# ------------------ END DEFENSE TRAINING -----------------------------

# ------------------ ATTACK -------------------------------------------
if response["response"] == "OK":
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
    print(response["samples"])
# ------------------ END ATTACK -----------------------------------------
