from .optimizer import BaseOptimizer
from tensors import Tensor

class SGD(BaseOptimizer):

    def __init__(self, model_parameters: list[Tensor], lr: float):

        self.params = model_parameters
        self.lr = lr

    def step(self):

        for param in self.params:
            param.data -= self.lr * param.grad

    def zero_grad(self):

        for param in self.params:
            param.reset_grad()