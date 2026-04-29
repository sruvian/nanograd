import numpy as np
from tensors import Tensor
from .optimizer import BaseOptimizer
from typing import Callable

class BFGS(BaseOptimizer):

    def __init__(self, model_parameters: list[Tensor]):
        super().__init__(model_parameters, lr = 1.0)
        n = sum(p.data.size for p in model_parameters)
        self.H = np.eye(n)  

    def _get_flat_params(self) -> np.ndarray:
        return np.concatenate([p.data.flatten() for p in self.params])

    def _get_flat_grads(self) -> np.ndarray:
        return np.concatenate([p.grad.flatten() for p in self.params])

    def _set_flat_params(self, flat_p: np.ndarray) -> None:
        idx = 0
        for p in self.params:
            size = p.data.size
            p.data = flat_p[idx:idx + size].reshape(p.data.shape)
            idx += size

    def line_search(self, flat_p: np.ndarray, direction: np.ndarray, closure: Callable[[], float], init_grad: np.ndarray, init_loss:float, c1:float = 1e-4)-> tuple[float, float, np.ndarray]:
        alpha = 1.0
        fin_loss = init_loss
        slope = init_grad @ direction
        fin_grad = init_grad.copy()

        for _ in range(20):
            self._set_flat_params(flat_p + alpha * direction)
            self.zero_grad()
            fin_loss = closure()
            fin_grad = self._get_flat_grads()

            armijo = fin_loss <= init_loss + c1 * alpha * slope

            if armijo:
                return alpha, fin_loss, fin_grad

            alpha *= 0.5

        self._set_flat_params(flat_p)
        self.zero_grad()
        closure()
        fin_grad = self._get_flat_grads()
        return 0.0, init_loss, fin_grad

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        
        assert closure is not None, "BFGS requires a closure"

        self._step_count = getattr(self, '_step_count', 0) + 1
        if self._step_count % 20 == 0:
            n = sum(p.data.size for p in self.params)
            self.H = np.eye(n)

        flat_p = self._get_flat_params()
        self.zero_grad()
        loss = closure()
        flat_g = self._get_flat_grads()
        direction = -self.H @ flat_g
        alpha, _, fin_grads = self.line_search(flat_p, direction, closure, flat_g, loss)
        fin_params = flat_p + alpha * direction
        self._set_flat_params(fin_params)

        s = fin_params - flat_p
        y = fin_grads - flat_g
        sy: float = float(y @ s)

        if sy > 1e-10:
            rho: float = 1.0 / sy
            I: np.ndarray = np.eye(len(s))
            A: np.ndarray = I - rho * np.outer(s, y)
            B: np.ndarray = I - rho * np.outer(y, s)
            self.H = A @ self.H @ B + rho * np.outer(s, s)

        return loss
