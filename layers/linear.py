from tensors import Tensor
import numpy as np
from .module import BaseModule


class Linear(BaseModule):

    def __init__(self, input_dim: int, output_dim: int, bias: bool = True)-> None:
        
        super().__init__()
        self.W = Tensor(np.random.rand(input_dim, output_dim))

        self.bias = bias
        if self.bias:
            self.b = Tensor(np.random.rand(1, output_dim))

    def forward(self, x: Tensor) -> Tensor:

        y = x @ self.W
        if self.bias:
            y = y+ self.b

        return y
    