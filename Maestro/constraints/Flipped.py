import numpy as np
class Flipped:
    '''
    Constraint that limits discrete flips of tokens/chars based on granuliarity
    Commonlhy used for attacks such as Hotflip.
    In this constraint, we assume the equal length of original sentence and perturbed sentence.
    We then compare each token and check for each token/char.
    '''
    def __init__(self,granularity:str,k:int):
        self.granularity = granularity
        self.k = k
    def violate(self,og_sentence, perturbed_sentence):
        diff = np.sum(og_sentence != perturbed_sentence)
        if diff > self.k:
            return True
        return False