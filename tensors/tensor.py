from __future__ import annotations
import numpy as np


class Tensor():

    def __init__(self, data: int | float | np.ndarray, _prev: tuple[Tensor, ...] = (), _op : str = "", req_grad: bool = True) -> None:
        
        self.data: np.ndarray = data if isinstance(data, np.ndarray) else np.array(data) # casting to NumPy for future Vectorization Support
        self.grad: np.ndarray = np.zeros_like(data)
        self.prev = _prev
        self.op = _op
        self._backward = lambda : None # Always attached to the child node
        self._req_grad = req_grad


    def __add__(self, other:int | float | Tensor) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(other)

        data = self.data + other.data
        out =  Tensor(data, _prev = (self, other), _op = "add")

        def _backward():
            if self._req_grad:
                self.grad += out.grad
            if other._req_grad:
                other.grad += out.grad
        out._backward = _backward

        return out
    
    def __radd__(self, other: int | float) -> Tensor: # Needed to support operations of type k + Tensor(b)
        return self + other

    def __sub__(self, other:int | float | Tensor) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(other)

        data = self.data - other.data
        out =  Tensor(data, _prev = (self, other), _op = "sub")
        def _backward():
            if self._req_grad:
                self.grad += out.grad
            if other._req_grad:
                other.grad += -out.grad
        out._backward = _backward
        
        return out
    
    def __rsub__(self, other: int | float) -> Tensor:
        return Tensor(other) - self


    def __mul__(self, other:int | float | Tensor) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(other)

        data = self.data * other.data
        out =  Tensor(data, _prev = (self, other), _op = "mul")

        def _backward():
             if self._req_grad:
                self.grad += out.grad * other.data
             if other._req_grad:
                other.grad += out.grad * self.data
        out._backward = _backward

        return out
    
    def __rmul__(self, other: int | float) -> Tensor:
        return self * other

    def __truediv__(self, other:int | float | Tensor) -> Tensor:
        if isinstance(other, (int, float)):
             other = Tensor(other)
        elif isinstance(other, Tensor):
             pass
        else:
             raise TypeError(" Cannot convert non int or float to Tensor")

        data = self.data / other.data
        out =  Tensor(data, _prev = (self, other), _op = "div")

        def _backward():
            if self._req_grad:
                self.grad += out.grad * (1 / other.data)
            if other._req_grad:
                other.grad += -out.grad * (self.data / (other.data ** 2))

        out._backward = _backward
        return out
    
    def __rtruediv__(self, other: int | float) -> Tensor:
        return Tensor(other) / self

    def __pow__(self, other:int | float | Tensor) -> Tensor:
        if isinstance(other, Tensor):
             if not isinstance(other.data, (int, float)):
                raise TypeError("Power Operation supports Tensor and Scalar only")
        if isinstance(other, (int, float)):
             other = Tensor(other, req_grad = False)
            

        data = self.data ** other.data
        out = Tensor(data, _prev = (self, other), _op = "pow")

        def _backward():
             if self._req_grad:
                self.grad += out.grad * other.data * (self.data ** (other.data-1))
        
        out._backward = _backward
        return out
    
    def backward(self):
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for parent in v.prev:
                    build(parent)
                topo.append(v)

        build(self)

        self.grad = np.ones_like(self.data)

        for v in reversed(topo):
            v._backward()

    def reset_grad(self) -> None:
         if self._req_grad:
            self.grad = np.zeros_like(self.data)



def log(x : Tensor) -> Tensor:
    # derivative of log(x) = 1/x
    if not isinstance(x, Tensor):
            x = Tensor(x)

    out =  Tensor(np.log(x.data), _prev = (x,), _op = "log" )
    def _backward():
         if x._req_grad:
            x.grad += out.grad * (1 / (x.data + 1e-8))
    out._backward = _backward
    return out

def exp(x : Tensor) -> Tensor:
    # derivative does not change
    if not isinstance(x, Tensor):
            x = Tensor(x)

    out = Tensor(np.exp(x.data), _prev = (x,), _op = "exp" )
    
    def _backward():
         if x._req_grad:
            x.grad += out.grad * out.data
    
    out._backward = _backward
    return out

def sin(x : Tensor) -> Tensor:
    # Derivative of sin(x) = cos(x)
    if not isinstance(x, Tensor):
            x = Tensor(x)

    out =  Tensor(np.sin(x.data), _prev = (x,), _op = "sin" )

    def _backward():
         if x._req_grad:
            x.grad += np.cos(x.data) * out.grad
    
    out._backward = _backward

    return out

def relu(x : Tensor) -> Tensor:
    # behaves like a linear function if greater than 0, else a constant.
    if not isinstance(x, Tensor):
            x = Tensor(x)

    out = Tensor(np.maximum(0, x.data), _prev = (x,), _op = "relu" )

    def _backward():
        if x._req_grad:
            mask = (x.data > 0).astype(x.grad.dtype)

            x.grad += out.grad * mask

    out._backward = _backward

    return out

def tanh(x: Tensor) -> Tensor:
    if not isinstance(x, Tensor):
        x = Tensor(x)
    
    out = Tensor(np.tanh(x.data), _prev = (x,), _op = "tanh")
    def _backward():
        if x._req_grad:

            x.grad += out.grad * (1 - out.data**2) # Hyperbolic Trig Identity
    out._backward = _backward

    return out