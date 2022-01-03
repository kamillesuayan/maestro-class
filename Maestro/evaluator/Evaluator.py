from Maestro.attacker_helper.attacker_request_helper import virtual_model
import importlib.util
from typing import List, Iterator, Dict, Tuple, Any, Type
import numpy as np
import torch
from torch.utils.data import DataLoader
from Maestro.attacker_helper.attacker_request_helper import virtual_model
from transformers.data.data_collator import default_data_collator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
from Maestro.data import get_dataset
from Maestro.models import build_model
from Maestro.pipeline import VisionPipeline

def load_attacker(application, student_id, student_name, task_folder, task, vm, attacker_path_list=None):
    print("load_attacker", attacker_path_list, task, application)
    if (task == "attack_homework")|(task == "attack_project"):
        print("playground\n")
        spec = importlib.util.spec_from_file_location(
        str(task_folder) + "_" + str(student_id),
        "../../playground/"
        + str(task)
        + "/"
        + str(task_folder)
        + "-"
        + str(student_id) + "-" + str(student_name)
        + ".py",
        )
    elif (task == "defense_homework")|(task == "defense_project"):
        attackers = []
        for attacker_path in attacker_path_list:
            spec = importlib.util.spec_from_file_location(str(task) + "_" + str(student_id), attacker_path)
            foo = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(foo)
            attacker = foo.GeneticAttack(vm, image_size=[1, 28, 28], temperature=0.1,n_generation=200,n_population=100,use_mask=False,mask_rate=0.3,step_size=0.1,mutate_rate=0.2,child_rate=0.2)
            attackers.append(attacker)
        return attackers
    else:
        print("load error")

    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    if task == "attack_homework":
        attacker = foo.GeneticAttack(
            vm, image_size=[1, 28, 28], temperature=0.1,n_generation=100,n_population=100,use_mask=False,mask_rate=0.3,step_size=0.1,mutate_rate=0.1,child_rate=0.2
        )
    elif task == "attack_project":
        attacker = foo.ProjectAttack(
            vm, image_size=[1, 28, 28], n_population=100, mutate_rate=0.2,
        )
    else:
        attacker = foo.GeneticAttack(
            vm, image_size=[1, 28, 28], n_population=100, mutate_rate=0.2,
        )
    return attacker

def load_defender(model_name,application, student_id, student_name, task_folder,task, device, spec_path = None):
    if spec_path:
        spec = importlib.util.spec_from_file_location(application + "_" + model_name, spec_path)
    else:
        spec = importlib.util.spec_from_file_location(
            str(application) + "_" + str(student_id),
            "../playground/"
            + str(task_folder)
            + "/"
            + str(application)
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

def load_pretrained_defender(model_name, application, task_folder,task,student_id, student_name,  device):
    model_path = (
        "../playground/"+ str(task_folder)+"/" + str(student_id) +"_group_project/lenet_defended_model.pth"
    )
    spec = importlib.util.spec_from_file_location(
        str(task) + "_" + str(student_id),
        "../playground/"+ str(task_folder)+ '/' +str(student_id) +"_group_project/"
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
        student_name,
        vm,
        task,
        app_pipeline=None,
        iterator_dataloader=None,
        constraint=None,
    ) -> None:
        self.app_pipeline = app_pipeline
        self.model_name = model_name
        self.student_id = student_id
        if task == "defense_homework":
            self.method = load_defender(model_name, application, student_id, student_name,task,task, self.app_pipeline.device)
        elif (task == "attack_homework") | (task == "attack_project"):
            self.method = load_attacker(application, student_id, student_name, task, task, vm)
        elif task == "defense_project":
            self.method = load_pretrained_defender(model_name, application, task, task,student_id, student_name, self.app_pipeline.device)
        elif "war" in task:
            if task == "war_attack":
                self.method = load_attacker(application, student_id, student_name, "war_phase", task, vm)
            elif task == "war_defend":
                self.method = load_pretrained_defender(model_name,application, "war_phase", task,student_id, student_name, self.app_pipeline.device)
        else:
            print("loading evaulator error")
        self.iterator_dataloader = iterator_dataloader
        self.vm = vm
        self.constraint = constraint
        self.task = task

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
    def _metrics_dict(self,score, final_acc, cost_time, distance,number_queries):
        return {"score":score,"final_acc":final_acc,"cost_time":cost_time,"distance":distance,"number_queries":number_queries}
    def _get_scores(self,start_time,final_acc,number_queries,distance,t_threshold = 600, q_threshold=6000,dis_threshold = 600):
        cost_time = time.perf_counter() - start_time
        if cost_time > t_threshold:
            time_score = 0
        else:
            time_score = 1 - cost_time/t_threshold

        if number_queries > q_threshold:
            query_score = 0
        else:
            query_score = 1 - number_queries/q_threshold

        if distance > dis_threshold:
            dis_score = 0
        else:
            dis_score = 1 - distance / dis_threshold
        print(cost_time, distance,number_queries)
        score = final_acc * 70 + query_score * 20 + dis_score * 10
        print("score: ", final_acc, query_score, dis_score,score)
        metrics = self._metrics_dict(score, final_acc, cost_time, distance,number_queries)
        return metrics

    def attack_one_batch(self, test_loader, attack_method, target_label, vm):
        # print(vm.application_name)
        distance = 0
        n_success_attack = 0

        og_images = []
        perturbed_images = []
        for batch in test_loader:
            # Call FGSM Attack
            labels = batch["labels"].cpu().detach().numpy()
            batch = batch["image"].cpu().detach().numpy()[0]  # [channel, n, n]
            og_images.append((labels[0],labels[0],np.squeeze(batch)))
            # print(labels.item(), labels.item())
            perturbed_data, perturbed_label, success = attack_method(
                batch, labels, target_label=target_label,
            )

            perturbed_images.append((labels[0],perturbed_label,np.squeeze(perturbed_data)))
            # print(batch.shape)
            # print(perturbed_data[0].shape)
            delta_data = batch - perturbed_data[0]
            distance += np.linalg.norm(delta_data)

            n_success_attack += success
        return distance, og_images, perturbed_images, n_success_attack

    def attack_evaluator(self, t_threshold = 600, q_threshold=18000, dis_threshold = 2000):
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
        print("start testing")
        # Loop over all examples in test set
        total_distance = 0
        total_n_success_attack = 0
        total_batch_output_count = 0
        total_batch_gradient_count = 0
        scores = []
        # for i in range(10):
        #     self.vm.batch_output_count = 0
        #     self.vm.batch_gradient_count = 0
        n_success_attack = 0
        test_loader = iterator_dataloader

        distance, og_images, perturbed_images, n_success_attack = self.attack_one_batch(test_loader, self.method.attack, target_label, self.vm)
            # exit(0)

        # visualization
        from Maestro.utils import visualize
        visualize(og_images, "before_GA.png")
        visualize(perturbed_images, "after_GA.png")

        # Calculate final accuracy for this epsilon
        final_acc = n_success_attack / float(len(test_loader))
        number_queries = self.vm.batch_output_count + self.vm.batch_gradient_count
        total_batch_output_count += self.vm.batch_output_count
        total_batch_gradient_count += self.vm.batch_gradient_count
        print(
            "target_label: {}\t Attack Success Rate = {} / {} = {}".format(
                target_label, n_success_attack, len(test_loader), final_acc))
        metrics = self._get_scores(start_time,final_acc,number_queries,distance,q_threshold=q_threshold,dis_threshold=dis_threshold)
        scores.append(metrics)
        total_distance += distance
        total_n_success_attack += n_success_attack

        # print("success rate")
        # print(total_n_success_attack/10,total_n_success_attack/(len(test_loader)*10.0))
        # print("# queries")
        # print(total_batch_output_count, total_batch_gradient_count)
        # print("time")
        # print(time.perf_counter() - start_time,(time.perf_counter() - start_time)/10)
        # print("distance")
        # print(total_distance/10)




        return metrics


    # def defense_evaluator_attacker_loader(self, applications=None):
    #     new_pipeline = VisionPipeline(
    #                 self.app_pipeline.scenario,
    #                 None,
    #                 None,
    #                 self.app_pipeline.validation_data.data,
    #                 self.method.model,
    #                 None,
    #                 self.app_pipeline.device,
    #                 None,
    #             )

    #     # applications["temp_war_defense_eval"] = new_pipeline
    #     return new_pipeline

    def defense_evaluator(self, IP_ADDR, PORT, model_name, applications=None, attacker_path_list=None):
        print(model_name, applications, attacker_path_list)
        trainset = self.app_pipeline.training_data.data
        testset = self.app_pipeline.validation_data.data
        dev_data = self.vm.get_data(data_type="test")
        targeted_dev_data = []
        target_label = 7
        for instance in dev_data:
            if instance["label"] != target_label:
                targeted_dev_data.append(instance)
        print(len(targeted_dev_data))
        test_loader = DataLoader(
            targeted_dev_data,
            batch_size=1,
            shuffle=True,
            collate_fn=default_data_collator,
        )
        # vm = virtual_model("http://"+IP_ADDR+":"+PORT, application_name="temp_war_defense_eval")

        # 1 if we want to evaluate adversarial examples, we need to first get the attacked examples
        attackers = load_attacker(None, self.student_id, self.task , self.task , self.vm, attacker_path_list)
        # print(attackers)
        for attacker in attackers:
            # print(attacker)
            distance, og_images, perturbed_images, n_success_attack = self.attack_one_batch(test_loader, attacker.attack, target_label, self.vm)

        adv_images = []
        gt_labels = []
        for i in perturbed_images:
            adv_images.append(i[2])
            gt_labels.append(i[0])
        # perturbed_images = np.array(perturbed_images)
        adv_images = torch.tensor(np.array(adv_images)).unsqueeze(0).type(torch.FloatTensor)
        gt_labels = torch.tensor(gt_labels)
        print(adv_images.shape, gt_labels.shape)
        adv_dataset = torch.utils.data.TensorDataset(adv_images, gt_labels)

        # print(perturbed_images.shape)
        # print(testset.shape)


        print("start adversarial training!")


        # 2 evaluate on adversarial data
        device = self.app_pipeline.device

        model = build_model(self.model_name, num_labels=None, max_length=None, device=device)
        model = self.method.train(model, trainset, device)
        model.eval()
        # adv data results
        adv_testloader = torch.utils.data.DataLoader(
            adv_dataset, batch_size=1, shuffle=True, num_workers=10
        )  # raw data
        # add adversarial data
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in adv_testloader:
                print(inputs.shape, labels.shape)
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(
            "Accuracy of the network on the images: %.3f %%" % (100 * correct / total)
        )
        adv_score = 100 * correct / total

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
                print(inputs.shape, labels.shape)

                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(
            "Accuracy of the network on the images: %.3f %%" % (100 * correct / total)
        )
        raw_score = 100 * correct / total
        score = (adv_score+raw_score)/2
        return {"score":score}

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
            attackers.append(load_attacker("temp_war_defense_eval", self.student_id, task, task,vm))
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
