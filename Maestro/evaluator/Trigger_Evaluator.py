import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class Trigger_Evaluator:
    def __init__(self, attacker, iterator_dataloader, vm, constraint) -> None:
        self.attacker = attacker
        self.iterator_dataloader = iterator_dataloader
        self.vm = vm
        self.constraint = constraint

    def evaluate_attacker(self):
        trigger = self.attacker.attack(self.iterator_dataloader, self.vm, constraint=3)
        constraint_violation = self._constraint(trigger)
        self._evaluate_with_trigger(trigger)
        print(f"Constraint Violation Cases: {constraint_violation}")

    def _evaluate_with_trigger(self, trigger):
        all_vals = []
        dataloader = self.iterator_dataloader
        trigger = torch.LongTensor(trigger)
        for batch in dataloader:
            # print("start_batch")
            # print(batch)
            flipped = [0]
            # get gradient w.r.t. trigger embeddings for current batch
            labels = batch["labels"]
            # print(batch["input_ids"])
            batch_trigger = trigger.repeat(batch["input_ids"].shape[0], 1)
            perturbed_tokens = torch.cat(
                (batch["input_ids"][:, :1], batch_trigger, batch["input_ids"][:, 1:],),
                1,
            )
            # print(perturbed_tokens)
            logits = self.vm.get_batch_output(
                perturbed_tokens.cpu().detach().numpy(), labels
            )
            # print(logits)
            preds = np.argmax(logits[1], axis=1)
            success = preds != labels
            all_vals.append(success)
        a = np.array(all_vals).mean()
        print(f"Label flip rate: {a}")

    def _constraint(self, trigger):
        size = len(trigger)
        if size > self.constraint:
            return True
        return False


