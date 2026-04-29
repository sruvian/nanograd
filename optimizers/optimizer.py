from tensors import Tensor
from typing import Callable

class BaseOptimizer:

    def __init__(self, model_parameters: list[Tensor], lr: float):

        self.params = model_parameters
        self.lr = lr

    def step(self, closure: Callable[[], float] | None = None) -> float | None:

        raise NotImplementedError

    def zero_grad(self):

        for param in self.params:
            param.reset_grad()