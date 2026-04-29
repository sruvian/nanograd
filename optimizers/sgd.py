from .optimizer import BaseOptimizer
from tensors import Tensor
from typing import Callable

class SGD(BaseOptimizer):

    def __init__(self, model_parameters: list[Tensor], lr: float):

        self.params = model_parameters
        self.lr = lr

    def step(self, closure: Callable[[], float] | None = None) -> float | None:

        for param in self.params:
            param.data -= self.lr * param.grad