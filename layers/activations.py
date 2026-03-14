from tensors import Tensor
import numpy as np
from tensors import tanh, relu, sigmoid
from .module import BaseModule

class ReLU(BaseModule):
    
    def __init__(self):
        super().__init__()
        self.relu = relu

    def forward(self, x):
        return self.relu(x)

class Tanh(BaseModule):
    
    def __init__(self):
        super().__init__()
        self.tanh = tanh
    
    def forward(self, x):
        return self.tanh(x)

class Sigmoid(BaseModule):
    
    def __init__(self):
        super().__init__()
        self.sigmoid = sigmoid

    def forward(self, x):
        return self.sigmoid(x)
