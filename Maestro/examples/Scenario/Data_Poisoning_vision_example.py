import os
from Maestro.data import HuggingFaceDataset, get_dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.data import DataLoader
from transformers.data.data_collator import default_data_collator
from Maestro.utils import move_to_device, get_embedding
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import copy
def get_accuracy(model, device, test_loader,c):
    model.eval()
    test_loss = 0
    correct = 0
    correct_class = 0
    with torch.no_grad():
        for batch in test_loader:
            # print(batch)
            data, target = batch["image"].to(device), batch["labels"].to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            correct_class += pred.view_as(target)[target==c].eq(target[target==c]).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print(f"Accuracy for class {c}: {100. * correct / len(test_loader.dataset)}%")
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
def main():
    device=1
    data = get_dataset("MNIST")
    print(data)
    train_data = data["train"].get_json_data()
    test_data = data["test"].get_json_data()
    print(len(train_data))
    test_loader = DataLoader(
        test_data,
        batch_size=32,
        shuffle=True,
        collate_fn=default_data_collator,
    )
    for i in train_data:
        i["image"] = torch.FloatTensor(i["image"])
    iterator_dataloader = DataLoader(
        train_data,
        batch_size=32,
        shuffle=True,
        collate_fn=default_data_collator,
    )

    C = 1#60
    K = 1
    nepochs = 1
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    perturbed_data = copy.deepcopy(train_data)
    train_dataloader = DataLoader(
            perturbed_data,
            batch_size=32,
            shuffle=True,
            collate_fn=default_data_collator,
        )

    og_class = 2
    target_class = 9
    class_data = [x for x in perturbed_data if x["label"] == og_class]
    adv_dataloader = DataLoader(
            class_data,
            batch_size=32,
            shuffle=True,
            collate_fn=default_data_collator,
        )

    poison_size = 25
    random_poisons = np.random.choice(range(len(perturbed_data)), poison_size,replace=False)
    print(random_poisons)
    for i in range(C):
        sample = None
        for k in range(K):
            optimizer.zero_grad()
            last = 0
            for idx,batch in enumerate(train_dataloader):
                data, target = batch["image"].to(device), batch["labels"].to(device)
                data.requires_grad = True
                last = data
                uids = batch["uid"]
                optimizer.zero_grad()
                sample = data
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
        ## update D_poison
        poison_data = []
        for z in perturbed_data:
            if z["uid"] in random_poisons:
                poison_data.append(z)
        print(len(poison_data))
        poison_loader = DataLoader(
            poison_data,
            batch_size=poison_size,
            shuffle=False,
            collate_fn=default_data_collator,
        )
        for z in poison_loader:
            data = z["image"].to(device)
            data.requires_grad = True
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True).squeeze()
            loss = F.nll_loss(output, pred)
            a = torch.autograd.grad(
                loss, last, create_graph=False, retain_graph=True
            )
            print(a)
            exit(0)
            loss.backward()
            x_grad = data.grad.data
        print(x_grad.shape)

        for z in range(poison_size):
            uid = random_poisons[z]
            perturbed_data[uid]["image"] = perturbed_data[uid]["image"] + 0.1 * x_grad[z].cpu().detach()

        
        train_dataloader = DataLoader(
            perturbed_data,
            batch_size=32,
            shuffle=True,
            collate_fn=default_data_collator,
        )
    print(train_data[0])



    model2 = Net().to(device)
    optimizer2 = optim.Adadelta(model2.parameters(), lr=1.0)
    scheduler2 = StepLR(optimizer2, step_size=1, gamma=0.7)
    perturbed_dataloader = DataLoader(
        perturbed_data,
        batch_size=32,
        shuffle=True,
        collate_fn=default_data_collator,
    )
    get_accuracy(model2,device,test_loader,2)
    for idx,batch in enumerate(perturbed_dataloader):
        data, target = batch["image"].to(device), batch["labels"].to(device)
        optimizer2.zero_grad()
        output = model2(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer2.step()
        
    get_accuracy(model2,device,test_loader,2)


if __name__ == "__main__":
    main()