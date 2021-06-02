import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# different types of constrains: discrete tokens, epsilon ball, chars, ...etc
class FGSM_Evaluator:
    def __init__(self, attacker, iterator_dataloader, vm, constraint) -> None:
        self.attacker = attacker
        self.iterator_dataloader = iterator_dataloader
        self.vm = vm
        self.constraint = constraint

    def evaluate_attacker(self):
        all_vals = []
        new_batches = []
        constraint_violations = 0
        for batch in self.iterator_dataloader:
            print("start_batch")
            # print(batch)
            flipped = [0]
            # get gradient w.r.t. trigger embeddings for current batch
            labels = batch["labels"].cpu().detach().numpy()
            perturbed = self.attacker.attack(
                batch["image"].cpu().detach().numpy(),
                labels,
                self.vm,
                self.constraint.epsilon,
            )
            logits = self.vm.get_batch_output(perturbed, labels)
            logits = np.array(logits)
            print(logits)
            preds = np.argmax(logits, axis=1)
            print(preds,labels)
            success = preds != labels
            print(success)
            constraint_violation = self._constraint(
                batch["image"].cpu().detach().numpy(), perturbed
            )
            constraint_violations += constraint_violation
            all_vals.append(success)
        a = np.array(all_vals).mean()
        print(f"label flip rate: {a}")
        print(f"Constraint Violation Cases: {constraint_violations}")

    def _constraint(self, original_input, perturbed_input):
        return self.constraint.violate(original_input,perturbed_input)