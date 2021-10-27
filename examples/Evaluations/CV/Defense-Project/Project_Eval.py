import importlib
import numpy as np
import sys
import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers.data.data_collator import default_data_collator

# ------------------ LOCAL IMPORTS ---------------------------------
from Maestro.data import TorchVisionDataset, get_dataset
from Maestro.evaluator import FGSM_Evaluator
from Maestro.models import build_model
from Maestro.defense_helper.defense_request_helper import virtual_model
from Maestro.constraints import Epsilon


class LENET(nn.Module):
    def __init__(self) -> None:
        super(LENET, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x) -> torch.Tensor:
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# ------------------  LOCAL IMPORTS --------------------------------
# sys.path.insert(0, "junlin_group_project/")
# module = importlib.import_module("project")

model_path = "junlin_group_project/lenet_defended_model.pth"
url = "http://127.0.0.1:5000"
model = LENET()
model.load_state_dict(torch.load(model_path, map_location="cpu"))
# ------------------ SPLIT DATASET --------------------------------
dataset_name = "MNIST"
datasets = get_dataset(dataset_name)

train_data = datasets["train"]
train_dataset = train_data.get_json_data()
print("train data", len(train_dataset), type(train_dataset))

test_data = datasets["test"]
test_dataset = test_data.get_json_data()
print("test data", len(test_dataset))
targeted_dev_data = test_dataset[:1000]
iterator_dataloader = DataLoader(
    targeted_dev_data, batch_size=32, collate_fn=default_data_collator,
)
# ------------------ END SPLIT DATASET --------------------------------

# ------------------ EVALUATION -------------------------------------------
acc = []
num = 0
adv_acc = 0.0
with torch.no_grad():
    for batch in iterator_dataloader:
        # print(batch)
        inputs = torch.FloatTensor(batch["image"])
        labels = batch["labels"]
        output = model(inputs)
        preds = torch.max(output, dim=1)[1].cpu().detach().numpy()
        # print(preds)
        success = preds == labels.cpu().detach().numpy()
        # print(success)
        acc.extend(success)

        # # use predicted label as target label
        # # with torch.enable_grad():
        # adv_data = self.attack.perturb(data, pred, "mean", False)
        # adv_output = model(adv_data, _eval=True)
        # adv_pred = torch.max(adv_output, dim=1)[1]
        # adv_acc = evaluate(adv_pred.cpu().numpy(), label.cpu().numpy(), "sum")
        # total_adv_acc += adv_acc
print(f"accuracy: {sum(acc)/len(acc)}")
