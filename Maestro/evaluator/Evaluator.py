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


class Evaluator:
    def __init__(self, application, student_id, vm, task,iterator_dataloader=None, constraint=None) -> None:
        if task == "defense_homework":
            self.method = self.load_defender(application, student_id, vm)
        elif ((task == "attack_homework") | (task == "attack_project")):
            self.method = self.load_attacker(application, student_id, task, vm)
        elif task == "defense_project":
            self.method = self.load_pretrained_defender(application, student_id, vm)
        else:
            print("loading evaulator error")
        self.iterator_dataloader = iterator_dataloader
        self.vm = vm
        self.constraint = constraint

    def load_attacker(self, application, student_id, task, vm):
        spec = importlib.util.spec_from_file_location(str(application)+"_"+str(student_id), "../tmp/"+str(task)+"/"+str(application)+"_"+str(student_id)+".py")
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)
        if task == "attack_homework":
            attacker = foo.GeneticAttack(vm, image_size=[1, 28, 28], n_population=100, mutate_rate=0.05,)
        else:
            attacker = foo.ProjectAttack(vm, image_size=[1, 28, 28], n_population=100, mutate_rate=0.05,)

        return attacker

    def load_defender(self, application, student_id, vm):
        spec = importlib.util.spec_from_file_location(str(application)+"_"+str(student_id),"../tmp/defense_homework/"+str(application)+"_"+str(student_id)+".py")
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)
        defender = foo.Data_Augmentation() # change to the defense class name
        return defender

    def load_pretrained_defender(self, application, student_id, vm):
        model_path = "../tmp/defense_project/junlin_group_project/lenet_defended_model.pth"
        url = "http://127.0.0.1:5000"
        spec = importlib.util.spec_from_file_location(str(application)+"_"+str(student_id),"./tmp/defense_project/junlin_group_project/"+str(application)+"_"+str(student_id)+".py")
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)
        model = foo.LENET()
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        defender = foo.Defender() # change to the defense class name
        return

    def attack_evaluator(self):
        start_time = time.clock()
        dataset_label_filter = 0
        target_label = 7
        dev_data = self.vm.get_data(data_type="test")
        print("attack_evaluator")
        targeted_dev_data = []
        for instance in dev_data:
            if instance["label"] == dataset_label_filter:
                targeted_dev_data.append(instance)
        print(len(targeted_dev_data))
        targeted_dev_data = targeted_dev_data[:10]
        universal_perturb_batch_size = 1
        # tokenizer = model_wrapper.get_tokenizer()
        iterator_dataloader = DataLoader(
            targeted_dev_data,
            batch_size=universal_perturb_batch_size,
            shuffle=True,
            collate_fn=default_data_collator,
        )
        print("started the process")
        all_vals = []
        n_success_attack = 0
        adv_examples = []

        print("start testing")
        # Loop over all examples in test set
        test_loader = iterator_dataloader
        distance = 0
        for batch in test_loader:
            # Call FGSM Attack
            labels = batch["labels"].cpu().detach().numpy()
            batch = batch["image"].cpu().detach().numpy()[0]  # [channel, n, n]
            print(labels.item(), labels.item())

            perturbed_data, success = self.method.attack(
                batch, labels, self.vm, target_label=target_label,
            )
            # print(batch.shape)
            # print(perturbed_data[0].shape)
            delta_data = batch - perturbed_data[0]
            distance += np.linalg.norm(delta_data)

            n_success_attack += success
            # exit(0)
        # Calculate final accuracy for this epsilon
        final_acc = n_success_attack / float(len(test_loader))
        print(
            "target_label: {}\t Attack Success Rate = {} / {} = {}".format(
                target_label, n_success_attack, len(test_loader), final_acc
            )
        )
        cost_time = time.clock() - start_time
        if cost_time > 100:
            time_score = 0
        else:
            time_score = 100/cost_time
        # print(final_acc, time_score, distance)
        score = final_acc * 70 + time_score * 0.20 + distance * 0.1
        return score


    def defense_evaluator(self, task):
        dataset_name = "MNIST"
        datasets = get_dataset(dataset_name)

        train_data = datasets["train"]
        train_dataset = train_data.get_json_data()
        print("train data", len(train_dataset), type(train_dataset))

        test_data = datasets["test"]
        test_dataset = test_data.get_json_data()
        print("test data", len(test_dataset))
        targeted_dev_data = test_dataset[:1000]
        iterator_dataloader = DataLoader(
            targeted_dev_data, batch_size=32, collate_fn=default_data_collator,
        )

        acc = []
        num = 0
        adv_acc = 0.0
        with torch.no_grad():
            for batch in iterator_dataloader:
                inputs = torch.FloatTensor(batch["image"])
                labels = batch["labels"]
                output = model(inputs)
                preds = torch.max(output, dim=1)[1].cpu().detach().numpy()
                # print(preds)
                success = preds == labels.cpu().detach().numpy()
                # print(success)
                acc.extend(success)

                # # use predicted label as target label
                # # with torch.enable_grad():
                # adv_data = self.attack.perturb(data, pred, "mean", False)
                # adv_output = model(adv_data, _eval=True)
                # adv_pred = torch.max(adv_output, dim=1)[1]
                # adv_acc = evaluate(adv_pred.cpu().numpy(), label.cpu().numpy(), "sum")
                # total_adv_acc += adv_acc
        print(f"accuracy: {sum(acc)/len(acc)}")
        score = sum(acc)/len(acc)
        return socre
