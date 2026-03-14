from layers import BaseModule
from tensors import Tensor

class MSE(BaseModule):
    
    def __init__(self, reduction = "mean"):
        reduction_list = { "sum": Tensor.sum, "mean": Tensor.mean, "max": Tensor.max}
        if reduction not in reduction_list:
            raise ValueError("Invalid reduction type")
        self.reduction = reduction_list[reduction]


    def forward(self, predictions, targets):

        return self.reduction(((targets - predictions)**2))
