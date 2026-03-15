from tensors import Tensor
import numpy as np


def vec_jac_prod(x: Tensor, y: Tensor) -> np.ndarray:
    flattened_y = y.reshape((-1,))
    J = []
    for i in range(flattened_y.shape[0]):
        zero_all_grads(flattened_y[i])
        x.grad = None
        flattened_y[i].backward()
        J.append(x.g.data.copy())
    return np.vstack(J)

def zero_all_grads(root: Tensor):
    visited = set()
    def walk(node):
        if id(node) in visited:
            return
        visited.add(id(node))
        node.grad = None
        for child in node.prev:
            if child.op != '':  # stop at forward graph leaves
                walk(child)
    walk(root)