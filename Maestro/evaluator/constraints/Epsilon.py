import numpy as np
class Epsilon:
    '''
    Constraint that limits contiuous epsilon ball around each pixel
    Commonlhy used for attacks such as FGSM.
    '''
    def __init__(self,epsilon:float,distance="l1"):
        self.epsilon = epsilon
        self.distance = distance
    def violate(self,original_input, perturbed_input):
        print(original_input.shape,perturbed_input.shape)
        assert original_input.shape == perturbed_input.shape
        
        diff = original_input - perturbed_input
        if self.distance == "l1":
            diff = np.abs(diff)
        violated_pixels = np.sum(diff > (self.epsilon+1e-5))

        if violated_pixels:
            return True
        return False
        


