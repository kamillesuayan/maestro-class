from Maestro.attacker_helper.attacker_request_helper import virtual_model
import importlib.util
from typing import List, Iterator, Dict, Tuple, Any, Type
import numpy as np
import torch
from torch.utils.data import DataLoader
from Maestro.attacker_helper.attacker_request_helper import virtual_model
from transformers.data.data_collator import default_data_collator
import matplotlib.pyplot as plt
import time
from Maestro.data import get_dataset
from Maestro.models import build_model
from Maestro.pipeline import VisionPipeline

def load_attacker(application, student_id, task_folder, task,vm):
    print("load_attacker")

    spec = importlib.util.spec_from_file_location(
        str(task) + "_" + str(student_id),
        "../tmp/"
        + str(task_folder)
        + "/"
        + str(application)
        + "_"
        + str(student_id)
        + ".py",
    )
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    if task == "attack_homework":
        attacker = foo.GeneticAttack(
            vm, image_size=[1, 28, 28], n_population=100, mutate_rate=0.2,
        )
    else:
        attacker = foo.ProjectAttack(
            vm, image_size=[1, 28, 28], n_population=100, mutate_rate=0.2,
        )
    return attacker

def load_defender(model_name,application, student_id, task_folder,task, device, spec_path = None):
    if spec_path:
        spec = importlib.util.spec_from_file_location(application + "_" + model_name, spec_path)
    else:
        spec = importlib.util.spec_from_file_location(
            str(task) + "_" + str(student_id),
            "../tmp/"
            + str(task_folder)
            + "/"
            + str(task)
            + "_"
            + str(student_id)
            + ".py",
        )
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    model = build_model(model_name, num_labels=None, max_length=None, device=device)
    defender = foo.Adv_Training(
            model,
            epsilon=0.2,
            alpha=0.1,
            min_val=0,
            max_val=1,
            max_iters=10,
            _type="linf",
        )
    return defender

def load_pretrained_defender(model_name, application, task_folder,task,student_id,  device):
    model_path = (
        "../tmp/"+ str(task_folder)+"/" + str(student_id) +"_group_project/lenet_defended_model.pth"
    )
    spec = importlib.util.spec_from_file_location(
        str(task) + "_" + str(student_id),
        "../tmp/"+ str(task_folder)+ '/' +str(student_id) +"_group_project/"
        + str(task)
        + "_"
        + str(student_id)
        + ".py",
    )
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    model = foo.LENET()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    defender = foo.ProjectDefense(
        model,
        epsilon=0.2,
        alpha=0.1,
        min_val=0,
        max_val=1,
        max_iters=10,
        _type="linf",
    )  # change to the defense class name
    return defender


class Evaluator:
    def __init__(
        self,
        application,
        model_name,
        student_id,
        vm,
        task,
        app_pipeline=None,
        iterator_dataloader=None,
        constraint=None,
    ) -> None:
        self.app_pipeline = app_pipeline
        self.model_name = model_name
        if task == "defense_homework":
            self.method = load_defender(model_name, application, student_id, task,task, self.app_pipeline.device)
        elif (task == "attack_homework") | (task == "attack_project"):
            self.method = load_attacker(application, student_id, task,task, vm)
        elif task == "defense_project":
            self.method = load_pretrained_defender(model_name,application, task, task,student_id, self.app_pipeline.device)
        elif "war" in task:
            if task == "war_attack":
                self.method = load_attacker(application, student_id,"war_phase", task, vm)
            elif task == "war_defend":
                self.method = load_pretrained_defender(model_name,application, "war_phase", task,student_id, self.app_pipeline.device)
        else:
            print("loading evaulator error")
        self.iterator_dataloader = iterator_dataloader
        self.vm = vm
        self.constraint = constraint

    # for the student debugging
    # def evaluate_attacker(self):
    #     all_vals = []
    #     new_batches = []
    #     constraint_violations = 0
    #     for batch in self.iterator_dataloader:
    #         print("start_batch")
    #         # print(batch)
    #         flipped = [0]
    #         # get gradient w.r.t. trigger embeddings for current batch
    #         labels = batch["labels"].cpu().detach().numpy()
    #         perturbed = self.method.attack(
    #             batch["image"].cpu().detach().numpy(),
    #             labels,
    #             self.vm,
    #             self.constraint.epsilon,
    #         )
    #         logits = self.vm.get_batch_output(perturbed, labels)
    #         logits = np.array(logits)
    #         print(logits)
    #         preds = np.argmax(logits, axis=1)
    #         print(preds, labels)
    #         success = preds != labels
    #         print(success)
    #         constraint_violation = self._constraint(
    #             batch["image"].cpu().detach().numpy(), perturbed
    #         )
    #         constraint_violations += constraint_violation
    #         all_vals.append(success)
    #     a = np.array(all_vals).mean()
    #     print(f"label flip rate: {a}")
    #     print(f"Constraint Violation Cases: {constraint_violations}")

    def _constraint(self, original_input, perturbed_input):
        return self.constraint.violate(original_input, perturbed_input)

    def attack_evaluator(self, t_threshold = 600, dis_threshold = 80):
        start_time = time.perf_counter()
        dataset_label_filter = 0
        target_label = 7
        dev_data = self.vm.get_data(data_type="test")
        print("attack_evaluator")
        targeted_dev_data = []
        for instance in dev_data:
            if instance["label"] != target_label:
                targeted_dev_data.append(instance)
        print(len(targeted_dev_data))
        targeted_dev_data = targeted_dev_data
        universal_perturb_batch_size = 1
        # tokenizer = model_wrapper.get_tokenizer()
        iterator_dataloader = DataLoader(
            targeted_dev_data,
            batch_size=universal_perturb_batch_size,
            shuffle=True,
            collate_fn=default_data_collator,
        )
        n_success_attack = 0
        print("start testing")
        # Loop over all examples in test set
        test_loader = iterator_dataloader
        distance = 0
        og_images = []
        perturbed_images = []
        for batch in test_loader:
            # Call FGSM Attack
            labels = batch["labels"].cpu().detach().numpy()
            batch = batch["image"].cpu().detach().numpy()[0]  # [channel, n, n]
            og_images.append((labels[0],labels[0],np.squeeze(batch)))
            # print(labels.item(), labels.item())

            perturbed_data, perturbed_label, success = self.method.attack(
                batch, labels, self.vm, target_label=target_label,
            )
            perturbed_images.append((labels[0],perturbed_label,np.squeeze(perturbed_data)))
            # print(batch.shape)
            # print(perturbed_data[0].shape)
            delta_data = batch - perturbed_data[0]
            distance += np.linalg.norm(delta_data)

            n_success_attack += success
            # exit(0)

        # visualization 
        from Maestro.utils import visualize
        visualize(og_images, "before_GA.png")
        visualize(perturbed_images, "after_GA.png")

        # Calculate final accuracy for this epsilon
        final_acc = n_success_attack / float(len(test_loader))
        print(
            "target_label: {}\t Attack Success Rate = {} / {} = {}".format(
                target_label, n_success_attack, len(test_loader), final_acc
            )
        )
        cost_time = time.perf_counter() - start_time

        if cost_time > t_threshold:
            time_score = 0
        else:
            time_score = 1 - cost_time/t_threshold
        #print(final_acc, time_score, distance)
        if distance > dis_threshold:
            dis_score = 0
        else:
            dis_score = 1 - distance / dis_threshold
        print(cost_time, distance)
        print("score: ", final_acc, time_score, dis_score)

        score = final_acc * 70 + time_score * 20 + dis_score * 10
        return score

    def defense_evaluator(self, model_name):
        trainset = self.app_pipeline.training_data.data
        device = self.app_pipeline.device

        model = build_model(self.model_name, num_labels=None, max_length=None, device=device)

        testset = self.app_pipeline.validation_data.data
        # if self.attacker is not None:
        #     testset = self.attacker.attack_dataset(testset, self.defender)
        model = self.method.train(model, trainset, device)
        model.eval()
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=True, num_workers=10
        )  # raw data
        # add adversarial data
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(
            "Accuracy of the network on the images: %.3f %%" % (100 * correct / total)
        )
        score = 100 * correct / total
        return score

    def defense_evaluator_project(self):
        device = self.app_pipeline.device
        model = self.method.model.to(device)

        testset = self.app_pipeline.validation_data.data
        # if self.attacker is not None:
        #     testset = self.attacker.attack_dataset(testset, self.defender)
        # model = self.method.train(model, trainset, device)
        model.eval()
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=True, num_workers=10
        )  # raw data
        # add adversarial data
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(
            "Accuracy of the network on the images: %.3f %%" % (100 * correct / total)
        )
        score = 100 * correct / total
        return score

    def defense_evaluator_war(self,applications=None, attacker_path_list=None):
        # trainset=self.app_pipeline.training_data.data
        device = self.app_pipeline.device
        testset = self.app_pipeline.validation_data.data
        model = self.method.model.to(device)
        new_pipeline = VisionPipeline(
                    None,
                    None,
                    None,
                    self.app_pipeline.validation_data.data,
                    self.method.model,
                    None,
                    self.app_pipeline.device,
                    None,
                )
        applications["temp_war_defense_eval"] = new_pipeline
        vm = virtual_model("http://"+IP_ADDR+":"+PORT, application_name="temp_war_defense_eval")
        attackers = []
        for i in range(len(attacker_path_list)):
            attackers.append(load_attacker("temp_war_defense_eval", student_id, task_folder, task,vm))
        print("start adversarial training!")
        # print(trainset.getitem())
        # print(len(trainset))

        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=True, num_workers=10
        )  # raw data
        # add adversarial data
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(
            "Accuracy of the network on the images: %.3f %%" % (100 * correct / total)
        )
        score = 100 * correct / total
        return score
