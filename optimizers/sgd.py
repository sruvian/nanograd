from .optimizer import BaseOptimizer
from tensors import Tensor
import numpy as np

class SGD(BaseOptimizer):

    def __init__(self, model_parameters: list[Tensor], lr: float):

        self.params = model_parameters
        self.lr = lr

    def step(self):

        for param in self.params:
            param.data -= self.lr * param.g.data

def zero_grad(self):
    for p in self.parameters():
        p.grad = Tensor(np.zeros_like(p.data))