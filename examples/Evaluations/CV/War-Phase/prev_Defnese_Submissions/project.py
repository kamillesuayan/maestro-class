import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from typing import List


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


def perturb(model, x_tensor, labels, epsilon, device=0):
    # --------------TODO--------------
    x_tensor = x_tensor.to(device)
    x_tensor.requires_grad = True
    output = model(x_tensor)
    pred = output.max(1, keepdim=True)[1]
    loss = F.nll_loss(output, pred.squeeze())
    model.zero_grad()
    loss.backward()
    x_grad = x_tensor.grad.data
    # x_grad = torch.FloatTensor(x_grad)
    sign_data_grad = x_grad.sign()
    perturbed_image = x_tensor + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # ------------END TODO-------------
    return perturbed_image.to(device)


def adversarial_training():
    defended_model_path = "lenet_defended_model.pth"
    device = 0
    target_net = LENET().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(target_net.parameters(), lr=0.001, momentum=0.9)
    transform = transforms.Compose([transforms.ToTensor()])
    batch_size = 32
    epsilon = 0.2

    trainset = torchvision.datasets.MNIST("./", transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    # testset = torchvision.datasets.MNIST("./", train=False)
    # testloader = torch.utils.data.DataLoader(
    #     testset, batch_size=batch_size, shuffle=False, num_workers=2
    # )
    running_loss = 0.0
    epochs = 10
    for e in range(epochs):
        for batch in trainloader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            perturbed_inputs = perturb(target_net, inputs, labels, epsilon, device)
            inputs = perturbed_inputs
            optimizer.zero_grad()
            outputs = target_net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    torch.save(target_net.state_dict(), defended_model_path)


if __name__ == "__main__":
    adversarial_training()

