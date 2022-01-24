from typing import List, Iterator, Dict, Tuple, Any, Type
import numpy as np
import torch
from torch.utils.data import DataLoader
from Maestro.attacker_helper.attacker_request_helper import virtual_model
from transformers.data.data_collator import default_data_collator
import matplotlib.pyplot as plt


class ProjectAttack:
    def __init__(
        self,
        vm,
        image_size: List[int],
        n_population=100,
        n_generation=100,
        mask_rate=0.2,
        temperature=0.3,
        use_mask=True,
        step_size = 0.1,
        child_rate = 0.5,
        mutate_rate = 0.6
    ):
        """
        args:
            vm: virtual model is wrapper used to get outputs/gradients of a model.
            image_size: [3,32,32]
            n_population: number of population in each iteration
            n_generation: maximum of generation constrained. The attack automatically stops when this maximum is reached
            mutate_rate: if use_mask is set to true, this is used to set the rate of masking when perturbed
            temperature: this sets the temperature when computing the probabilities for next generation
            use_mask: when this is true, only a subset of the image will be perturbed.

        """
        self.vm = vm
        self.image_size = image_size
        self.n_population = n_population
        self.n_generation = n_generation
        self.mask_rate = mask_rate
        self.temperature = temperature
        self.use_mask= use_mask
        self.step_size = step_size
        self.child_rate = child_rate
        self.mutate_rate = mutate_rate

    def attack(
        self,
        original_image:  np.ndarray,
        labels: List[int],
        target_label: int,
    ):
        """
        currently this attack has 2 versions, 1 with no mask pre-defined, 1 with mask pre-defined.
        args:
            original_image: a numpy ndarray images, [3,32,32]
            labels: label of the image, a list of size 1
            target_label: target label we want the image to be classified, int
        return:
            the perturbed image
            label of that perturbed iamge
            success: whether the attack succeds
        """
        self.original_image = np.array(original_image)
        self.mask = np.random.binomial(1, self.mask_rate, size=self.image_size).astype("bool")
        population = self.init_population(original_image)
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
                # visualize([(labels[0],np.argmax(output[best_index, :]),np.squeeze(population[best_index]))], "after_GA1.png")
                success = True
                break
        return [population[best_index]], np.argmax(output[best_index, :]),success

    def fitness(self, image: np.ndarray, target: int):
        """
        evaluate how fit the current image is
        return:
            output: output of the model
            scores: the "fitness" of the image, measured as logits of the target label
        """
        output = self.vm.get_batch_output(image)
        scores = output[:, target]
        return output, scores

    def fitness2(self, image: np.ndarray, target: int):
        """
        evaluate how fit the current image is
        return:
            output: output of the model
            scores: the "fitness" of the image, measured as logits of the target label
        """
        output = self.vm.get_batch_output(image)
        scores = output[:, target]
        mse = np.sqrt(np.power(image - self.original_image, 2).sum(axis=-1).sum(axis=-1).mean(axis=-1))
        return output, scores-mse*0.3

    def eval_population(self, population, target_label):
        """
        evaluate the population, pick the parents, and then crossover to get the next
        population
        args:
            population: current population, a list of images
            target_label: target label we want the imageto be classiied, int
        return:
            population: population of all the images
            output: output of the model
            scores: the "fitness" of the image, measured as logits of the target label
            best_indx: index of the best image in the population
        """

        print("POPULATION SHAPE:")
        input()
        output, scores = self.fitness(population, target_label)
        # output, scores = self.fitness2(population, target_label)
        # --------------TODO--------------
        score_ranks = np.sort(scores)[::-1]  # Sort the scores from largest to smallest
        best_index = np.argmax(scores)
        logits = np.exp(scores/self.temperature)  # Exponentiate the scores after incorporating temperature
        select_probs = logits / np.sum(logits) # Normalize the logits between 0-1
        # ------------END TODO-------------

        if np.argmax(output[best_index, :]) == target_label:
            return population, output, scores, best_index

        # --------------TODO--------------
        # the elite gene that's definitely in the next population without perturbation
        elite = [population[best_index]]
        # strong and fit genes passed down to next generation, they have a chance to mutate
        survived = []  # Survived, and mutate some of them

        survive_amt = int(self.n_population * (1-self.child_rate))
        for i in range(1, survive_amt):
            if i < survive_amt // 2: # keep the best half survivors
                next_best = np.where(scores == score_ranks[i])[0][0]
                survived.append(population[next_best])
            else: # perturb other half of survivors
                next_best = np.where(scores == score_ranks[i])[0][0]
                survived.append(self.perturb(population[next_best]))

        # offsprings of strong genes
        children = []

        # choose parents based on their probabilities
        for _ in range(self.n_population - (len(elite) + len(survived))):
            # choose random index based on probabilities
            # parent = np.random.choice(self.n_population, 2, replace=False, p=select_probs)
            # children.append(self.crossover(population[parent[0]], population[parent[1]]))
            parent = np.random.choice(self.n_population, 1, replace=False, p=select_probs)
            children.append(self.crossover(population[parent[0]], population[best_index]))

        # population =np.array(elite + survived +childs)
        population = population
        # ------------END TODO-------------
        return population, output, scores, best_index

    def perturb(self, image):
        """
        perturb a single image with some constraints and a mask
        args:
            image: the image to be perturbed
        return:
            perturbed: perturbed image
        """
        if not self.use_mask:
            perturbed = np.clip(image + np.random.randn(*self.mask.shape) * self.step_size, 0, 1)
        else:
            perturbed = np.clip(
                image + self.mask * np.random.randn(*self.mask.shape) * self.step_size, 0, 1
            )

        return perturbed

    def crossover(self, x1, x2):
        """
        crossover two images to get a new one. We use a uniform distribution with p=0.5
        args:
            x1: image #1
            x2: image #2
        return:
            x_new: newly crossovered image
        """
        x_new = x1.copy()
        for i in range(len(x1)):
            for j in range(len(x1[i])):
                if np.random.uniform() < 0.5:
                    x_new[0][i][j] = x2[0][i][j]
        return x_new

    def init_population(self, original_image: np.ndarray):
        """
        Initialize the population to n_population of images. Make sure to perturbe each image.
        args:
            original_image: image to be attacked
        return:
            a list of perturbed images initialized from orignal_image
        """
        return np.array([self.perturb(original_image[0]) for _ in range(self.n_population)])


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


