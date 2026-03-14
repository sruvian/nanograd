from __future__ import annotations
import numpy as np


class Tensor():

    def __init__(self, data: int | float | np.ndarray | list, _prev: tuple[Tensor, ...] = (), _op : str = "", req_grad: bool = True) -> None:
        
        self.data: np.ndarray = data if isinstance(data, np.ndarray) else np.array(data, dtype = float) # casting to NumPy for future Vectorization Support
        self.grad: np.ndarray = np.zeros_like(self.data, dtype = float)
        self.prev = _prev
        self.op = _op
        self._backward = lambda : None # Always attached to the child node
        self._req_grad = req_grad
        self.shape = self.data.shape


    def __add__(self, other:int | float | Tensor) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(other)

        data = self.data + other.data
        out =  Tensor(data, _prev = (self, other), _op = "add")

        def _backward():
            if self._req_grad:
                self.grad += unbroadcast(out.grad, self.data.shape)
            if other._req_grad:
                other.grad += unbroadcast(out.grad, other.data.shape)

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
                self.grad += unbroadcast(out.grad, self.data.shape)
            if other._req_grad:
                other.grad += -unbroadcast(out.grad, other.data.shape)
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
                self.grad += unbroadcast(out.grad * other.data, self.data.shape)
             if other._req_grad:
                other.grad += unbroadcast(out.grad * self.data, other.data.shape)
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
                self.grad += unbroadcast(out.grad, self.data.shape) * (1 / other.data)
            if other._req_grad:
                other.grad += - unbroadcast(out.grad, other.data.shape) * (self.data / (other.data ** 2))

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

    def __matmul__(self: Tensor, other: Tensor) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out = Tensor(np.matmul(self.data, other.data), _prev = (self, other), _op = "matmul")

        def _backward():
            if self._req_grad:
                self.grad += out.grad @ other.data.T
            if other._req_grad:
                other.grad += self.data.T @ out.grad
        
        out._backward = _backward
        return out
    
    def __getitem__(self, idx) -> Tensor:
        
        out = Tensor(self.data[idx], _prev= (self, ), _op = "slice")

        def _backward():
            if self._req_grad:
                self.grad[idx] += out.grad
        out._backward = _backward
        return out
    
    
    


    def sum(self: Tensor) -> Tensor:
        out = Tensor(np.sum(self.data), _prev = (self,), _op = "sum")

        def _backward():
            if self._req_grad:
                self.grad += np.ones_like(self.data) * out.grad

        out._backward = _backward

        return out
    
    def mean(self: Tensor) -> Tensor:
        
        return self.sum() / self.data.size
    
    def max(self: Tensor) -> Tensor:

        idx = np.unravel_index(np.argmax(self.data), self.data.shape)
        out = Tensor(np.max(self.data), _prev = (self,), _op = "max")

        def _backward():
        
            self.grad[idx] += out.grad
        
        out._backward = _backward

        return out

    def reshape(self: Tensor, shape: int| tuple) -> Tensor:

        out = Tensor(np.reshape(self.data, shape), _prev = (self, ), _op = "reshape")

        def _backward():
            if self._req_grad:
                self.grad += np.reshape(out.grad, self.data.shape)
        
        out._backward = _backward

        return out
    
    @property
    def T(self: Tensor) -> Tensor:

        out = Tensor(self.data.T, _prev = (self, ), _op = "T")

        def _backward():
            if self._req_grad:
                self.grad += out.grad.T
        
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

def unbroadcast(grad: np.ndarray, shape) -> np.ndarray:

    while grad.ndim > len(shape):
        grad = grad.sum(axis = 0)

    for i, dim in enumerate(shape):
        if dim == 1:
            grad = grad.sum(axis = i, keepdims = True)

    return grad

def concat(tensor_list: list[Tensor] | tuple[Tensor, ...], axis: int = 0):

    data = np.concatenate([t.data for t in tensor_list], axis=axis)

    out = Tensor(data, _prev=tuple(tensor_list), _op="concat")

    def _backward():
        if out.grad is None:
            return

        start = 0
        for t in tensor_list:
            if not t._req_grad:
                start += t.data.shape[axis]
                continue

            size = t.data.shape[axis]

            slicer = [slice(None)] * out.grad.ndim
            slicer[axis] = slice(start, start + size)

            t.grad += out.grad[tuple(slicer)]

            start += size

    out._backward = _backward

    return out


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

def sigmoid(x: Tensor) -> Tensor:

    if not isinstance(x, Tensor):
        x = Tensor(x)
    
    temp = 1 / (1 + np.exp(-x.data))
    out = Tensor(temp, _prev = (x, ), _op = "sigmoid")

    def _backward():
        if x._req_grad:
            x.grad += out.grad * out.data * (1 - out.data)
    
    out._backward = _backward

    return out