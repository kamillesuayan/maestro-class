import torch
from typing import List, Iterator, Dict, Tuple, Any, Type, Union, cast
import torch.nn as nn
import torch.nn.functional as F
import transformers
import math

def build_model(model_name, num_labels: int, max_length: int = 128, device: int = 0, pretrained_file: str = None):
    if model_name == "LeNet":
        model = LeNet()
    elif model_name == "VGG11":
        model = vgg11()
    elif model_name == "LeNet_Mnist":
        model = LeNet_Mnist()
    # model = model.to(device)
    if pretrained_file != None:
        model.load_state_dict(torch.load(pretrained_file, map_location=device))
        # model.load_state_dict(torch.load(pretrained_file))
    return model

# LeNet
class LeNet_Mnist(nn.Module):
   def __init__(self) -> None:
       super(LeNet_Mnist, self).__init__()
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

# CIFAR 10 used in maestro-class
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x) -> torch.tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
