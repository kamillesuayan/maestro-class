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
        mask_rate=0.2,
        temperature=0.3,
        use_mask=True,
    ):
        """
        args:
            vm: virtual model is wrapper used to get outputs/gradients of a model. 
            image_size: [1,28,28]
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

    def attack(
        self,
        original_image:  np.ndarray,
        labels: List[int],
        target_label: int,
    ):
        """
        currently this attack has 2 versions, 1 with no mask pre-defined, 1 with mask pre-defined.
        args:
            original_image: a numpy ndarray images, [1,28,28]
            labels: label of the image, a list of size 1
            target_label: target label we want the image to be classified, int
        return:
            the perturbed image
            label of that perturbed iamge
            success: whether the attack succeds
        """
        self.original_image = original_image
        self.mask = np.random.binomial(1, self.mask_rate, size=self.image_size).astype("bool")
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
        args: 
            image: the image to be perturbed
        return:
            perturbed: perturbed image
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



# def main():
#     # (1) prepare the data loaders and the model
#     server_url = "http://128.195.56.136:5000"  # used when the student needs to debug on the server
#     local_url = "http://127.0.0.1:5000"  # used when the student needs to debug locally

#     vm = virtual_model(local_url, application_name="FGSM")
#     dataset_label_filter = 0
#     target_label = 7
#     dev_data = vm.get_data(data_type="test")

#     targeted_dev_data = []
#     for instance in dev_data:
#         if instance["label"] == dataset_label_filter:
#             targeted_dev_data.append(instance)
#     print(len(targeted_dev_data))
#     targeted_dev_data = targeted_dev_data[:10]
#     universal_perturb_batch_size = 1
#     # tokenizer = model_wrapper.get_tokenizer()
#     iterator_dataloader = DataLoader(
#         targeted_dev_data,
#         batch_size=universal_perturb_batch_size,
#         shuffle=True,
#         collate_fn=default_data_collator,
#     )
#     print("started the process")
#     all_vals = []
#     n_success_attack = 0
#     adv_examples = []

#     GA = GeneticAttack(vm, image_size=[1, 28, 28], n_population=100, mutate_rate=0.05,)

#     print("start testing")
#     # Loop over all examples in test set
#     test_loader = iterator_dataloader
#     for batch in test_loader:
#         # Call FGSM Attack
#         labels = batch["labels"].cpu().detach().numpy()
#         batch = batch["image"].cpu().detach().numpy()[0]  # [channel, n, n]
#         print(labels.item(), labels.item())
#         visualize([(labels.item(), labels.item(), np.squeeze(batch))], "before.png")
#         perturbed_data, success = GA.attack(
#             batch, labels, vm, target_label=target_label,
#         )
#         visualize(
#             [(labels.item(), target_label, np.squeeze(perturbed_data))], "after.png"
#         )
#         n_success_attack += success
#         exit(0)
#     # Calculate final accuracy for this epsilon
#     final_acc = n_success_attack / float(len(test_loader))
#     print(
#         "target_label: {}\t Attack Success Rate = {} / {} = {}".format(
#             target_label, n_success_attack, len(test_loader), final_acc
#         )
#     )


if __name__ == "__main__":
    main()
