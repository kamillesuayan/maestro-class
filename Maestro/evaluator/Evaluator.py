class Evaluator:
    def __init__(self, attacker, iterator_dataloader, vm, constraint) -> None:
        self.attacker = attacker
        self.iterator_dataloader = iterator_dataloader
        self.vm = vm
        self.constraint = constraint
