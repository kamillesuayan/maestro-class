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
        mutate_rate=0.2,
        temperature=0.3,
        use_mask=True,
    ):
        self.vm = vm
        self.image_size = image_size
        self.n_population = n_population
        self.n_generation = n_generation
        self.mutate_rate = mutate_rate
        self.temperature = temperature
        self.use_mask= use_mask

    def attack(
        self,
        original_image: np.ndarray,
        labels: List[int],
        vm: virtual_model,
        target_label: int,
    ):
        """
        currently this attack has 2 versions, 1 with no mask pre-defined, 1 with mask pre-defined.
        """
        self.original_image = original_image
        self.mask = np.random.binomial(1, self.mutate_rate, size=self.image_size).astype("bool")
        population = self.init_population(original_image)
        print(len(population))
        examples = [(labels[0], labels[0], np.squeeze(x)) for x in population[:10]]
        visualize(examples, "population.png")
        success = False
        for g in range(self.n_generation):
            population, output, scores, best_index = self.eval_population(
                population, target_label
            )
            print(f"Generation: {g} best score: {scores[best_index]}")
            if np.argmax(output[best_index, :]) == target_label:
                print(f"Attack Success!")
                visualize([(labels[0],np.argmax(output[best_index, :]),np.squeeze(population[best_index]))], "after_GA1.png")
                success = True
                break
        return [population[best_index]], np.argmax(output[best_index, :]),success

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

    def perturb(self, image):
        """
        perturb a single image with some constraints and a mask
        """
        if not self.use_mask:
            # --------------TODO--------------
            perturbed = np.clip(image + np.random.randn(*self.mask.shape) * 0.1, 0, 1)
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
