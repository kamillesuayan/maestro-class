import importlib.util
from typing import List, Iterator, Dict, Tuple, Any, Type
from torch.utils.data import DataLoader
import time
import requests
import pickle
from utils import *
from pathlib import Path
import datetime

def load_defender(task_folder, task, device):
    spec = importlib.util.spec_from_file_location(
        str(task) ,
        "tasks/"
        + str(task_folder)
        + "/train.py",
    )
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    defender = foo.Adv_Training(device)
    return defender

class Evaluator:
    def __init__(self, task, dataset, file_path, device=None) -> None:
        self.device = device
        self.dataset = dataset
        self.file_path = file_path
        if (task == "defense_project"):
            self.method = load_defender(task, task, self.device)
        else:
            print("method load error.")
        self.model = self.method.model
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

    def defense_evaluator(self, dataset_path, attacker_path_list=None, pretrained=False):
        print("defense_evaluator: ", attacker_path_list, self.task)
        trainset = self.dataset['train']
        valset = self.dataset['val']
        testset = self.dataset['test'] # TorchVisionDataset
        # testset = self.dataset['server_test'] # TorchVisionDataset

        if pretrained == False:
            print("start adversarial training!")
            model = self.method.train(self.method.model, trainset, valset, self.device)
            torch.save(model.state_dict(), "models/defense_project-model.pth")
        else:
            model = self.method.model

        spec = importlib.util.spec_from_file_location( str(self.task),
            "tasks/" + str(self.task) + "/predict.py")
        predict_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(predict_module)
        defender_predict = predict_module.Prediction(self.device, self.file_path)
        self.model = defender_predict.model

        model.eval()
        # 1 if we want to evaluate adversarial examples, we need to first get the attacked examples
        print("dataset_path: ", dataset_path)
        adv_test_server_path = os.path.join(dataset_path, "adv_test_server_split.pt")
        if False: #os.path.exists(adv_test_server_path):
            print("load existed test server dataset!")
            adv_images_list = torch.load(adv_test_server_path)['adv_images_list']
            gt_labels_list = torch.load(adv_test_server_path)['gt_labels_list']
        else:
            attackers = self.load_attacker(self.task, self.task, attacker_path_list)
            adv_images_list = []
            gt_labels_list = []
            n_success_attack_list = []

            for attacker in attackers:
                adv_images = []
                gt_labels = []
                (distance, perturbed_images, n_success_attack) = self.adv_generator(testset, attacker.attack)
                n_success_attack_list.append(n_success_attack)
                for image in perturbed_images:
                    adv_images.append(image[1])
                    gt_labels.append(image[0])
                adv_images = (
                    torch.tensor(np.array(adv_images)).type(torch.FloatTensor)
                )
                gt_labels = torch.tensor(gt_labels)
                adv_images_list.append(adv_images)
                gt_labels_list.append(gt_labels)
                # print(adv_images.shape, gt_labels.shape)
            # print(n_success_attack_list)
            # adv_dataset = torch.utils.data.TensorDataset(adv_images, gt_labels)
            # list_length = len(attack)
            torch.save({
                "adv_images_list": adv_images_list,
                "gt_labels_list": gt_labels_list
            }, adv_test_server_path)



        # evaluate on the adv examples
        adv_accs = []
        # print(attacker_path_list)
        for i in range(len(attacker_path_list)):
            adv_dataset = torch.utils.data.TensorDataset(adv_images_list[i], gt_labels_list[i])

            adv_testloader = torch.utils.data.DataLoader(
                adv_dataset, batch_size=100, shuffle=True, num_workers=10
            )  # raw data
            # add adversarial data
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in adv_testloader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    # outputs = model(inputs)
                    predicted = defender_predict.batch_predict(inputs)
                    # _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    correct += (predicted == torch.full(predicted.shape, -1, dtype=torch.int).to(self.device)).sum().item()
            print("Accuracy of the network on the adv images: %.3f %%" % (100 * correct / total))
            adv_acc = 100 * correct / total
            adv_accs.append(adv_acc)
        # print(adv_accs)

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
                predicted = defender_predict.batch_predict(inputs)

                # outputs = model(inputs)
                # _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print("Accuracy of the network on the images: %.3f %%" % (100 * correct / total))
        raw_acc = 100 * correct / total
        return {"adv_accs": adv_accs, "raw_acc": raw_acc}


    def load_attacker(self, task_folder, task, attacker_path_list=None):
        if (task == "defense_project"):
            attackers = []
            for attacker_path in attacker_path_list:
                spec = importlib.util.spec_from_file_location(
                    str(task), attacker_path
                )
                foo = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(foo)
                attacker = foo.Attack(self.model, self.device)
                attackers.append(attacker)
            return attackers
        else:
            print("load error")
        return

def main():
    students_submission_path = "tasks/defense_project/"
    task = "defense_project"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_configs = {
                "name": "CIFAR10",
                "dataset_path": "datasets/CIFAR10/",
                "student_train_number": 10000,
                "server_train_number": 10000,
                "student_val_number": 1000,
                "student_test_number": 1000,
                "server_test_number": 1000
    }
    dataset = get_dataset("CIFAR10", dataset_configs)
    file_path = 'models/'
    evaluator = Evaluator(task=task, dataset=dataset, file_path=file_path, device=device)
    attacker_path_list = ["tasks/defense_project/attacker_list/attack_FGSM.py", "tasks/defense_project/attacker_list/attack_PGD.py"]
    metrics = evaluator.defense_evaluator("datasets/CIFAR10/", attacker_path_list, pretrained=False)
    print(metrics)

if __name__ == "__main__":
    main()
