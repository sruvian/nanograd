from .jacobian import zero_all_grads
import numpy as np
from tensors import Tensor

def hess(func: Tensor, x: Tensor) -> np.ndarray:
    func.backward()
    n = x.data.shape[0]
    H = np.zeros((n, n))
    
    # slice each component of x.grad — these are graph B nodes
    grad_components = [x.grad[i] for i in range(n)]
    
    for i, g_i in enumerate(grad_components):
        # zero all graph B nodes but NOT x itself
        zero_all_grads(g_i)
        x.grad = None  # clear x.grad so accumulate_grad assigns fresh
        g_i.backward()
        if x.grad is not None:
            H[i] = x.grad.data.copy()
        # else H[i] stays zeros (no second derivative)
    
    return H
