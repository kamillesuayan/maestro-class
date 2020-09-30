from pipeline import Pipeline,Scenario, Attacker
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# class model_wrapper:
#     def __init__(self, model, scenario,application):
#         self.model = model
#         self.scenario = scenario
#         self.application = application
#     def get_input_gradient(self,x):
#         if self.scenario.attacker.output_access == False:
#             raise ValueError("Do not have access to gradients")
#         device = self.application.device
#         x= x.to(device)
#         x.requires_grad = True
#         output = self.model(x)
#         pred = output.max(1, keepdim=True)[1]
#         loss = F.nll_loss(output, pred[0])
#         self.model.zero_grad()
#         loss.backward()
#         x_grad = x.grad.data
#         return x_grad
#     def get_prediction(self,x):
#         if self.scenario.attacker.output_access == False:
#             raise ValueError("Do not have access to predictions")
#         device = self.application.device
#         x = x.to(device)
#         output = self.model(x)
#         init_pred = output.max(1, keepdim=True)[1]
#         return init_pred

class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
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
        print(x)
        exit(0)
        return F.log_softmax(x, dim=1)
def fgsm_attack(image, epsilon, data_grad) -> torch.Tensor:
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon*sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image
def test(model_wrapper, device, test_loader, epsilon ):

    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in test_loader:
        data = data.to(device)
        output = model_wrapper.get_output(data)
        init_pred = output.max(1, keepdim=True)[1]
        print(init_pred,target)
        if init_pred.item() != target.item():
            continue
        data_grad = model_wrapper.get_input_gradient(data)
        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model_wrapper.get_output(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


def main():
    # (1) prepare the data loaders and the model
    pretrained_model = "data/lenet_mnist_model.pth"
    use_cuda=True
    test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=1, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=1, shuffle=True)
    test_data = test_loader
    training_data = train_loader
    dev_data =None
    print("CUDA Available: ",torch.cuda.is_available())
    device = torch.device("cuda:1" if (use_cuda and torch.cuda.is_available()) else "cpu")
    model = Net().to(device)
    model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))
    model.eval()

    training_process = None
    # (2) initialize Atacker, which specifies access rights
    training_data_access = False
    dev_data_access = True
    test_data_access = True
    model_access = False
    output_access = 2
    myattacker = Attacker(training_data_access,dev_data_access,test_data_access,model_access,output_access)

    # (3) initialize Scenario. This defines our target
    target = None
    myscenario = Scenario(target,myattacker)

    model_wrapper = Pipeline(myscenario,training_data,dev_data,test_data,model,training_process,device).get_object()

    # (4) test FGSM
    test(model_wrapper, device, test_loader, 0)

if __name__ == "__main__":
    main()