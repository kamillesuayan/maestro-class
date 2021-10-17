from Maestro.attacker_helper.attacker_request_helper import virtual_model
import importlib.util
from typing import List, Iterator, Dict, Tuple, Any, Type
import numpy as np
import torch
from torch.utils.data import DataLoader
from Maestro.attacker_helper.attacker_request_helper import virtual_model
from transformers.data.data_collator import default_data_collator
import matplotlib.pyplot as plt

class Evaluator:
    def __init__(self, application, student_id, vm, iterator_dataloader=None, constraint=None) -> None:
        self.attacker = self.load_attacker(application, student_id, vm)
        self.iterator_dataloader = iterator_dataloader
        self.vm = vm
        self.constraint = constraint

    def load_attacker(self, application, student_id, vm):
        spec = importlib.util.spec_from_file_location(str(application)+"_"+str(student_id),"../tmp/attack_homework/"+str(application)+"_"+str(student_id)+".py")
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)
        attacker = foo.GeneticAttack(vm, image_size=[1, 28, 28], n_population=100, mutate_rate=0.05,)
        return attacker

    def attack_evaluator(self):
        dataset_label_filter = 0
        target_label = 7
        dev_data = self.vm.get_data(data_type="test")

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
        for batch in test_loader:
            # Call FGSM Attack
            labels = batch["labels"].cpu().detach().numpy()
            batch = batch["image"].cpu().detach().numpy()[0]  # [channel, n, n]
            print(labels.item(), labels.item())

            perturbed_data, success = self.attacker.attack(
                batch, labels, self.vm, target_label=target_label,
            )

            n_success_attack += success
            # exit(0)
        # Calculate final accuracy for this epsilon
        final_acc = n_success_attack / float(len(test_loader))
        print(
            "target_label: {}\t Attack Success Rate = {} / {} = {}".format(
                target_label, n_success_attack, len(test_loader), final_acc
            )
        )
        score = final_acc
        return score


    def defense_evaluator():
        return socre
