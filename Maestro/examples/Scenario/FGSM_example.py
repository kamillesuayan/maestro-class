import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import sys

from Maestro.pipeline import Pipeline, Scenario, Attacker
from Maestro.models.model import build_model
from Maestro.data.DataModifier import get_data


def fgsm_attack(image, epsilon, data_grad) -> torch.Tensor:
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


def test(model_wrapper, device, epsilon):

    # Accuracy counter
    correct = 0
    adv_examples = []
    print("start testing")
    # Loop over all examples in test set
    test_loader = model_wrapper.test_data.get_write_data()
    for data, target in test_loader:
        data = data.to(device)
        output = model_wrapper.get_output(data)
        init_pred = output.max(1, keepdim=True)[1]
        print(init_pred, target)
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
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    # Calculate final accuracy for this epsilon
    final_acc = correct / float(len(test_loader))
    print(
        "Epsilon: {}\tTest Accuracy = {} / {} = {}".format(
            epsilon, correct, len(test_loader), final_acc
        )
    )

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


def main():
    # (1) prepare the data loaders and the model
    pretrained_model = "data/lenet_mnist_model.pth"
    use_cuda = True
    train_loader, dev_loader, test_loader, vocab = get_data("MNIST")

    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device(
        "cuda:1" if (use_cuda and torch.cuda.is_available()) else "cpu"
    )
    name = "FGSM_example_model"
    model = build_model(name, pretrained_model).to(device)
    model.eval()

    training_process = None
    # (2) initialize Atacker, which specifies access rights
    attacker_config = "FGSM.yaml"
    myattacker = Attacker()
    myattacker.load_from_yaml(attacker_config)

    # (3) initialize Scenario. This defines our target
    target = None
    myscenario = Scenario(target, myattacker)

    model_wrapper = Pipeline(
        myscenario,
        train_loader,
        dev_loader,
        test_loader,
        model,
        training_process,
        device,
    ).get_object()

    # (4) test FGSM
    test(model_wrapper, device, 0.1)


if __name__ == "__main__":
    main()
