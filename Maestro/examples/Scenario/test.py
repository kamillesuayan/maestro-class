import torch
import torch.nn as nn


class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()
        self.embedding = torch.nn.Embedding(21, 128, padding_idx=0)
        self.fc1 = nn.Linear(128, 10)
        self.fc2 = nn.Linear(10, 128)
        self.tanh = nn.Tanh()

    def forward(self, x):
        embed = self.embedding(x)
        a = torch.sum(embed, dim=0)
        x = self.fc1(a)
        # print(f"fc1: \n{x}")
        x = self.tanh(x)
        # print(f"tanh: \n{x}")
        x = self.fc2(x)
        # print(f"fc2: \n{x,a}")
        return x, a


# def criterior(a, b):
#     print(a.shape)
#     print(b.shape)
#     print((a - b).shape)
#     return nn.MSELoss(a, b)
#     # return torch.sum((a - b) ** 2)


random_ix = torch.LongTensor([8, 8, 8, 4, 8, 7])
model = myModel()

dataset = torch.randint(low=0, high=20, size=(1000, 8))
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters())
criterior = nn.MSELoss()
i = 0
for t in range(10000):
    sentence = dataset[i]
    pred, a = model(sentence)
    loss = criterior(a, pred)
    if t % 100 == 99:
        print(t, loss.item())
        print(a, pred)
    i += 1
    if i >= 1000:
        i = 0
    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()
