import importlib.util
from typing import List, Iterator, Dict, Tuple, Any, Type
from torch.utils.data import DataLoader
import time
from models import build_model
import requests
import pickle
from utils import *

def load_defender(model_name, student_id, name, task_folder, task, device, pretrained_file="models/lenet_mnist_model.pth"):
    spec = importlib.util.spec_from_file_location(
        str(task) + "_" + str(student_id),
        "tasks/"
        + str(task_folder)
        + "/"
        + str(task)
        + ".py",
    )
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    model = build_model(model_name, num_labels=None, max_length=None, device=device, pretrained_file=pretrained_file)
    defender = foo.Adv_Training(model, device, epsilon=0.2, min_val=0, max_val=1)
    return defender

class Evaluator:
    def __init__(self, model_name, student_id, name, task, dataset, device=None, pretrain_path="models/lenet_mnist_model.pth") -> None:
        self.model_name = model_name
        self.student_id = student_id
        self.device = device
        self.dataset = dataset
        if (task == "defense_homework"):
            self.method = load_defender(model_name, student_id, name, task, task, self.device, pretrain_path)
        else:
            print("method load error.")
        self.task = task

    def adv_generator(self, testset, attack_method):
        distance = []
        n_success_attack = 0
        perturbed_images = []
        for img, labels in testset:
            # print(img.shape)
            perturbed_data, success = attack_method(img, [labels])
            perturbed_images.append((labels, np.squeeze(perturbed_data)))
            delta_data = img - perturbed_data[0]
            distance.append(np.linalg.norm(delta_data))
            n_success_attack += success
        return distance, perturbed_images, n_success_attack

    def defense_evaluator(self, model_name, dataset_path, attacker_path_list=None, pretrained=False):
        print("defense_evaluator: ", model_name, attacker_path_list, self.task)
        trainset = self.dataset['train']
        testset = self.dataset['test'] # TorchVisionDataset

        if pretrained == False:
            print("start adversarial training!")
            model = self.method.train(self.method.model, trainset, self.device)
            torch.save(model.state_dict(), "models/defense_homework-model.pth")
        model.eval()

        # 1 if we want to evaluate adversarial examples, we need to first get the attacked examples
        # print("dataset_path: ", dataset_path)
        adv_test_server_path = os.path.join(dataset_path, "adv_test_server_split.pt")
        if os.path.exists(adv_test_server_path):
            print("load existed test server dataset!")
            adv_dataset = torch.load(adv_test_server_path)
        else:
            attackers = self.load_attacker(self.student_id, None, self.task, self.task, attacker_path_list)
            adv_images = []
            gt_labels = []
            for attacker in attackers:
                (distance, perturbed_images, n_success_attack) = self.adv_generator(testset, attacker.attack)

                for image in perturbed_images:
                    adv_images.append(image[1])
                    gt_labels.append(image[0])
            adv_images = (torch.tensor(np.array(adv_images)).unsqueeze(1).type(torch.FloatTensor))
            gt_labels = torch.tensor(gt_labels)
            # print(adv_images.shape, gt_labels.shape)
            # print(n_success_attack)
            adv_dataset = torch.utils.data.TensorDataset(adv_images, gt_labels)
            torch.save(adv_dataset, adv_test_server_path)
        # print("adv_dataset: ", len(adv_dataset))
        # evaluate on the adv examples
        adv_testloader = torch.utils.data.DataLoader(
            adv_dataset, batch_size=1, shuffle=True, num_workers=10
        )  # raw data
        # add adversarial data
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in adv_testloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        # print(correct)
        print("Accuracy of the network on the adv images: %.3f %%" % (100 * correct / total))
        adv_acc = 100 * correct / total

        # return score

        # 3 evaluate on original data
        # raw data result
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=True, num_workers=10
        )  # raw data

        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                # print(inputs.shape, labels.shape)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print("Accuracy of the network on the images: %.3f %%" % (100 * correct / total))
        raw_acc = 100 * correct / total
        return {"adv_acc": adv_acc, "raw_acc": raw_acc}


    def load_attacker(self, student_id, name, task_folder, task, attacker_path_list=None):
        if (task == "defense_homework"):
            attackers = []
            for attacker_path in attacker_path_list:
                spec = importlib.util.spec_from_file_location(
                    str(task) + "_" + str(student_id), attacker_path
                )
                foo = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(foo)

                attacker = foo.FGSMAttack(self.method.model, self.device)
                attackers.append(attacker)
            return attackers
        else:
            print("load error")
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)
        return attacker


def main():
    model_name = "LeNet_Mnist"
    student_id = "11"
    name = "Alice"
    task = "defense_homework"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_configs = {
                "name": "MNIST",
                "dataset_path": "datasets/MNIST/",
                "student_train_number": 10000,
                "server_train_number": 10000,
                "student_test_number": 1000,
                "server_test_number": 1000
    }
    dataset = get_dataset("MNIST", dataset_configs)
    # pretrain_path = "models/defense_homework-model"

    evaluator = Evaluator(model_name, student_id, name, task, dataset, device)
    attacker_path_list = ["tasks/defense_homework/attacker_list/attack_1.py"]
    metrics = evaluator.defense_evaluator(model_name, "datasets/MNIST/", attacker_path_list, pretrained=False)
    print(metrics)
if __name__ == "__main__":
    main()
