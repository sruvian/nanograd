from __future__ import annotations
import numpy as np


class Tensor():

    def __init__(self, data: int | float | np.ndarray | list, _prev: tuple[Tensor, ...] = (), _op : str = "", req_grad: bool = True) -> None:
        
        self.data: np.ndarray = data if isinstance(data, np.ndarray) else np.array(data, dtype = float) # casting to NumPy for future Vectorization Support
        self.grad: Tensor | None = None
        self.prev = _prev
        self.op = _op
        self._backward = lambda : None # Always attached to the child node
        self._req_grad = req_grad
        self.shape = self.data.shape
        self.ndim = len(self.shape)

    @property
    def g(self) -> Tensor:
        assert self.grad is not None, f"grad is None on '{self.op}' node"
        return self.grad

    def accumulate_grad(self, grad:Tensor):

        if self.grad is None:
            self.grad = grad
        else:
            self.grad = self.grad + grad


    def __add__(self, other:int | float | Tensor) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(other)

        data = self.data + other.data
        out =  Tensor(data, _prev = (self, other), _op = "add")

        def _backward():
            if self._req_grad:
                self.accumulate_grad(unbroadcast(out.g, self.data.shape))
            if other._req_grad:
                other.accumulate_grad(unbroadcast(out.g, other.data.shape))

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
                self.accumulate_grad(unbroadcast(out.g, self.data.shape))
            if other._req_grad:
                other.accumulate_grad(-unbroadcast(out.g, other.data.shape))
        out._backward = _backward
        
        return out
    
    def __rsub__(self, other: int | float) -> Tensor:
        return Tensor(other) - self

    def __neg__(self) -> Tensor:
        out = Tensor(-self.data, _prev=(self,), _op="neg")
        def _backward():
            if self._req_grad:
                self.accumulate_grad(out.g * Tensor(-1.0, req_grad=False))
        out._backward = _backward
        return out

    def __mul__(self, other:int | float | Tensor) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(other)

        data = self.data * other.data
        out =  Tensor(data, _prev = (self, other), _op = "mul")

        def _backward():
             if self._req_grad:
                self.accumulate_grad(unbroadcast(out.g * other, self.data.shape))
             if other._req_grad:
                other.accumulate_grad(unbroadcast(out.g * self, other.data.shape))
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
                self.accumulate_grad(unbroadcast(out.g, self.data.shape) * (1 / other))
            if other._req_grad:
                other.accumulate_grad(unbroadcast(out.g, other.data.shape) * (-(self / (other ** 2))))

        out._backward = _backward
        return out
    
    def __rtruediv__(self, other: int | float) -> Tensor:
        return Tensor(other) / self

    def __pow__(self, other:int | float | Tensor) -> Tensor:
        if isinstance(other, Tensor):
            if other.data.ndim != 0:  # only allow scalar tensors as exponents
                raise TypeError("Power operation supports scalar exponents only")
        if isinstance(other, (int, float)):
             other = Tensor(other, req_grad = False)
            

        data = self.data ** other.data
        out = Tensor(data, _prev = (self, other), _op = "pow")

        def _backward():
             if self._req_grad:
                exp = Tensor(other.data - 1, req_grad=False)
                self.accumulate_grad(out.g * other * (self ** exp))
        
        out._backward = _backward
        return out

    def __matmul__(self: Tensor, other: Tensor) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out = Tensor(np.matmul(self.data, other.data), _prev = (self, other), _op = "matmul")

        def _backward():
            if self._req_grad:
                self.accumulate_grad(out.g @ other.T)
            if other._req_grad:
                other.accumulate_grad(self.T @ out.g)
        
        out._backward = _backward
        return out
    
    def __getitem__(self, idx) -> Tensor:
        out = Tensor(self.data[idx], _prev=(self,), _op="slice")
        def _backward():
            if self._req_grad:
                is_scalar_idx = isinstance(idx, (int, np.integer)) or (
                    isinstance(idx, tuple) and all(isinstance(i, (int, np.integer)) for i in idx)
                )
                if is_scalar_idx:
                    # scalar index: mask approach keeps graph B connected
                    mask = np.zeros_like(self.data)
                    mask[idx] = 1.0
                    mask_tensor = Tensor(mask, req_grad=False)
                    self.accumulate_grad(mask_tensor * out.g)
                else:
                    # slice index: numpy scatter, first order only
                    grad_data = np.zeros_like(self.data)
                    grad_data[idx] = out.g.data
                    self.accumulate_grad(Tensor(grad_data, req_grad=False))
        out._backward = _backward
        return out
        
    def sum(self, axis: int | tuple | None = None, keepdims: bool = False) -> Tensor:
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), _prev=(self,), _op="sum")

        def _backward():
            if self._req_grad:
                grad = out.g
                # if we reduced an axis without keepdims, we need to restore the
                # dimension so numpy can broadcast it back to self.data.shape
                if axis is not None and not keepdims:
                    grad = grad.reshape(np.expand_dims(out.g.data, axis=axis).shape)
                self.accumulate_grad(grad * Tensor(np.ones_like(self.data)))

        out._backward = _backward
        return out
    
    def mean(self: Tensor) -> Tensor:
        
        return self.sum() / self.data.size
    
    def max(self: Tensor) -> Tensor:

        idx = np.unravel_index(np.argmax(self.data), self.data.shape)
        out = Tensor(np.max(self.data), _prev = (self,), _op = "max")

        def _backward():
        
            if self._req_grad:
                grad_data = np.zeros_like(self.data)
                grad_data[idx] = out.g.data
                self.accumulate_grad(Tensor(grad_data))
        
        out._backward = _backward

        return out

    def reshape(self: Tensor, shape: tuple) -> Tensor:

        if isinstance(shape, int):
            shape = (shape, )
        out = Tensor(np.reshape(self.data, shape), _prev = (self, ), _op = "reshape")

        def _backward():
            if self._req_grad:
                self.accumulate_grad(out.g.reshape(self.data.shape))
        
        out._backward = _backward

        return out
    
    @property
    def T(self: Tensor) -> Tensor:

        out = Tensor(self.data.T, _prev = (self, ), _op = "T")

        def _backward():
            if self._req_grad:
                self.accumulate_grad(out.g.T)
        
        out._backward = _backward

        return out


    def backward(self, grad: Tensor | None = None):

        if grad is None:
            grad = Tensor(np.ones_like(self.data))

        topo = []
        visited = set()
        self.grad = grad    
        def build(v):
            if v not in visited:
                visited.add(v)
                for parent in v.prev:
                    build(parent)
                topo.append(v)

        build(self)


        for v in reversed(topo):
            v._backward()

    def reset_grad(self) -> None:
         if self._req_grad:
            self.grad = None

def unbroadcast(grad: Tensor, shape: tuple) -> Tensor:
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad

def concat(tensor_list: list[Tensor] | tuple[Tensor, ...], axis: int = 0):

    data = np.concatenate([t.data for t in tensor_list], axis=axis)

    out = Tensor(data, _prev=tuple(tensor_list), _op="concat")

    def _backward():
        start = 0
        for t in tensor_list:
            if not t._req_grad:
                start += t.data.shape[axis]
                continue
            size = t.data.shape[axis]
            slicer = [slice(None)] * out.g.ndim
            slicer[axis] = slice(start, start + size)
            t.accumulate_grad(out.g[tuple(slicer)])
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
            x.accumulate_grad(out.g * (1 / (x + 1e-8)))
    out._backward = _backward
    return out

def exp(x : Tensor) -> Tensor:
    # derivative does not change
    if not isinstance(x, Tensor):
            x = Tensor(x)

    out = Tensor(np.exp(x.data), _prev = (x,), _op = "exp" )
    
    def _backward():
         if x._req_grad:
            x.accumulate_grad(out.g * out)
    
    out._backward = _backward
    return out

def sin(x : Tensor) -> Tensor:
    # Derivative of sin(x) = cos(x)
    if not isinstance(x, Tensor):
            x = Tensor(x)

    out =  Tensor(np.sin(x.data), _prev = (x,), _op = "sin" )

    def _backward():
         if x._req_grad:
            x.accumulate_grad(Tensor(np.cos(x.data)) * out.g)
    
    out._backward = _backward

    return out

def relu(x : Tensor) -> Tensor:
    # behaves like a linear function if greater than 0, else a constant.
    if not isinstance(x, Tensor):
            x = Tensor(x)

    out = Tensor(np.maximum(0, x.data), _prev = (x,), _op = "relu" )

    def _backward():
        if x._req_grad:
            mask = Tensor((x.data > 0).astype(x.data.dtype), req_grad=False)
            x.accumulate_grad(out.g * mask)

    out._backward = _backward

    return out

def tanh(x: Tensor) -> Tensor:
    if not isinstance(x, Tensor):
        x = Tensor(x)
    
    out = Tensor(np.tanh(x.data), _prev = (x,), _op = "tanh")
    def _backward():
        if x._req_grad:

            x.accumulate_grad(out.g * (1 - out**2)) # Hyperbolic Trig Identity
    out._backward = _backward

    return out

def sigmoid(x: Tensor) -> Tensor:

    if not isinstance(x, Tensor):
        x = Tensor(x)
    
    temp = 1 / (1 + np.exp(-x.data))
    out = Tensor(temp, _prev = (x, ), _op = "sigmoid")

    def _backward():
        if x._req_grad:
            x.accumulate_grad(out.g * out * (1 - out))
    
    out._backward = _backward

    return out