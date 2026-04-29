from re import L
from typing import Callable

from tensors import Tensor
import numpy as np

def numerical_hessian(f:Callable , x: Tensor, eps: float = 1e-4) -> np.ndarray:
    n = x.data.size
    H = np.zeros((n, n))

    for i in range(n):
        # perturb +eps
        x.data.flat[i] += eps
        x.reset_grad()
        out = f()
        out.backward()
        g_plus = x.grad.flatten()

        # perturb -2eps (to get from +eps to -eps)
        x.data.flat[i] -= 2 * eps
        x.reset_grad()
        out = f()
        out.backward()
        g_minus = x.grad.flatten()

        # restore
        x.data.flat[i] += eps

        H[i] = (g_plus - g_minus) / (2 * eps)

    return H