from typing import List, Iterator, Dict, Tuple, Any, Type
import numpy as np
import torch
from torch.utils.data import DataLoader
from Maestro.attacker_helper.attacker_request_helper import virtual_model
from transformers.data.data_collator import default_data_collator
import matplotlib.pyplot as plt


class GeneticAttack:
    def __init__(
        self,
        vm,
        image_size: List[int],
        n_population=100,
        n_generation=100,
        mutate_rate=0.05,
        temperature=0.3,
    ):
        self.vm = vm
        self.image_size = image_size
        self.n_population = n_population
        self.n_generation = n_generation
        self.mutate_rate = mutate_rate
        self.temperature = temperature

    def attack(
        self,
        original_image: List[List[int]],
        labels: List[int],
        vm: virtual_model,
        target_label: int,
    ):
        """
        currently this attack has 2 versions, 1 with no mask pre-defined, 1 with mask pre-defined.
        """
        self.original_image = original_image
        self.mask = np.random.binomial(1, 0.2, size=self.image_size).astype("bool")
        population = self.init_population(original_image)
        print(len(population))
        examples = [(0, 0, np.squeeze(x)) for x in population[:10]]
        visualize(examples, "population.png")
        for g in range(self.n_generation):
            success = False
            population, output, scores, best_index = self.eval_population(
                population, target_label
            )
            print(f"Generation: {g} best score: {scores[best_index]}")
            if np.argmax(output[best_index, :]) == target_label:
                print(f"Attack Success!")
                success = True
                break

        return [population[best_index]], success

    def fitness(self, image: List[List[int]], target: int):
        output = self.vm.get_batch_output(image)
        scores = output[:, target]
        return output, scores

    def eval_population(self, population, target_label):
        # --------------TODO--------------
        output, scores = self.fitness(population, target_label)
        logits = np.exp(scores / self.temperature)
        select_probs = logits / np.sum(logits)
        score_ranks = np.argsort(scores)[::-1]
        best_index = score_ranks[0]

        if np.argmax(output[best_index, :]) == target_label:
            return population, output, scores, best_index
        elite = [population[best_index]]
        mom_index = np.random.choice(
            self.n_population, self.n_population - 1, p=select_probs
        )
        dad_index = np.random.choice(
            self.n_population, self.n_population - 1, p=select_probs
        )
        childs = [
            self.crossover(population[mom_index[i]], population[dad_index[i]])
            for i in range(self.n_population - 1)
        ]
        childs = [self.perturb(childs[i]) for i in range(len(childs))]
        population = elite + childs
        # ------------END TODO-------------
        return population, output, scores, best_index

    def perturb(self, image, mask=True):
        """
        perturb a single image with some constraints and a mask 
        """
        if not mask:
            # --------------TODO--------------
            mask = np.random.binomial(1, 0.1, size=self.image_size).astype("bool")
            perturbed = np.clip(image + np.random.randn(*mask.shape) * 0.1, 0, 1)
            # ------------END TODO-------------
        else:
            # --------------TODO--------------
            perturbed = np.clip(
                image + self.mask * np.random.randn(*self.mask.shape) * 0.1, 0, 1
            )
            # ------------END TODO-------------

        return perturbed

    def crossover(self, x1, x2):
        x_new = x1.copy()
        for i in range(len(x1)):
            for j in range(len(x1[i])):
                if np.random.uniform() < 0.5:
                    x_new[0][i][j] = x2[0][i][j]
        return x_new

    def init_population(self, original_image: List[List[int]]):
        return [self.perturb(original_image[0]) for _ in range(self.n_population)]


def visualize(examples, filename):
    cnt = 0
    plt.figure(figsize=(8, 10))
    for j in range(len(examples)):
        cnt += 1
        plt.subplot(1, len(examples), cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        orig, adv, ex = examples[j]
        plt.title("{} -> {}".format(orig, adv))
        plt.imshow(ex, cmap="gray")
    plt.tight_layout()
    plt.show()
    plt.savefig(filename)


def main():
    # (1) prepare the data loaders and the model
    server_url = "http://128.195.56.136:5000"  # used when the student needs to debug on the server
    local_url = "http://127.0.0.1:5000"  # used when the student needs to debug locally

    vm = virtual_model(local_url, application_name="FGSM")
    dataset_label_filter = 0
    target_label = 7
    dev_data = vm.get_data(data_type="test")

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

    GA = GeneticAttack(vm, image_size=[1, 28, 28], n_population=100, mutate_rate=0.05,)

    print("start testing")
    # Loop over all examples in test set
    test_loader = iterator_dataloader
    for batch in test_loader:
        # Call FGSM Attack
        labels = batch["labels"].cpu().detach().numpy()
        batch = batch["image"].cpu().detach().numpy()[0]  # [channel, n, n]
        print(labels.item(), labels.item())
        visualize([(labels.item(), labels.item(), np.squeeze(batch))], "before.png")
        perturbed_data, success = GA.attack(
            batch, labels, vm, target_label=target_label,
        )
        visualize(
            [(labels.item(), target_label, np.squeeze(perturbed_data))], "after.png"
        )
        n_success_attack += success
        exit(0)
    # Calculate final accuracy for this epsilon
    final_acc = n_success_attack / float(len(test_loader))
    print(
        "target_label: {}\t Attack Success Rate = {} / {} = {}".format(
            target_label, n_success_attack, len(test_loader), final_acc
        )
    )


if __name__ == "__main__":
    main()
