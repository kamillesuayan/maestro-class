from typing import List, Iterator, Dict, Tuple, Any, Type
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers.data.data_collator import default_data_collator

# ------------------ LOCAL IMPORTS ---------------------------------
from Maestro.attacker_helper.attacker_request_helper import virtual_model
# ------------------ LOCAL IMPORTS ---------------------------------

def attack(
    original_tokens: List[List[int]],
    labels: List[int],
    vm: virtual_model,
    epsilon: float,
):
    # --------------TODO--------------
    data_grad = vm.get_batch_input_gradient(original_tokens, labels)
    data_grad = torch.FloatTensor(data_grad)
    sign_data_grad = data_grad.sign()
    perturbed_image = torch.FloatTensor(original_tokens) + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # ------------END TODO-------------
    return perturbed_image.cpu().detach().numpy()


class Detector(nn.Module):
    # --------------TODO--------------
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    # ------------END TODO-------------


def detector(train_set):
    model = Detector()
    model.train()
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True, num_workers=10)
    dataset_size = len(trainset)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(self.model.parameters())
    for epoch in range(epoches):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / dataset_size))
        running_loss = 0.0
    return model
