# nanograd

nanograd is a minimal deep learning framework implementing reverse-mode automatic differentiation from scratch in Python.

The project builds a small neural network library on top of a custom autodiff engine, supporting tensor operations, neural network layers, and gradient-based optimization including second-order methods.

The goal is to understand the core mechanics behind modern ML frameworks by implementing them from first principles.

---

## Features

### Automatic Differentiation
- Reverse-mode autodiff (backpropagation)
- Computational graph construction via Python closures
- Topological backward traversal
- Gradient accumulation for reused nodes

### Tensor Operations
- Broadcasting support
- Elementwise arithmetic (`+`, `-`, `*`, `/`, `pow`)
- Matrix multiplication
- Reshape and transpose
- Tensor slicing and indexing
- Reductions (`sum`, `mean`, `max`)

### Neural Network Components
- Linear layers
- Activation functions: ReLU, Tanh, Sigmoid
- Sequential model container
- Loss functions: Mean Squared Error (MSE)

### Convolution Layers
- Conv1D (sliding window convolution)
- Conv2D (im2col + matrix multiplication)

### Optimization
- Stochastic Gradient Descent (SGD)
- BFGS with Armijo line search (quasi-Newton second-order method)

### Second-Order Analysis
- Numerical Hessian via central finite differences

---

## Example

Training a small neural network:

```python
from tensors import Tensor
from layers import Linear, Sequential, Tanh
from losses import MSE
from optim import SGD

model = Sequential(
    Linear(2, 4),
    Tanh(),
    Linear(4, 1)
)

criterion = MSE()
optimizer = SGD(model.parameters(), lr=0.1)

x = Tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = Tensor([[0.], [1.], [1.], [0.]])

for _ in range(2000):
    pred = model(x)
    loss = criterion(y, pred)
    model.zero_grad()
    loss.backward()
    optimizer.step()

print(pred.data)
```

BFGS solves the same problem in ~40 steps:

```python
from optim import BFGS

optimizer = BFGS(model.parameters())

def closure():
    optimizer.zero_grad()
    pred = model(x)
    loss = criterion(y, pred)
    loss.backward()
    return float(loss.data)

for _ in range(100):
    optimizer.step(closure)
```

```
## Project Structure
nanograd/
│
├── tensors/
│   └── tensor.py         # core autodiff engine
│
├── layers/
│   ├── module.py
│   ├── linear.py
│   ├── conv.py
│   └── activations.py
│
├── losses/
│   └── mse.py
│
├── optim/
│   ├── optimizer.py      # base class
│   ├── sgd.py
│   └── bfgs.py
│
├── functional/
│   └── hessian.py        # numerical Hessian via central differences
│
├── tests/
│   ├── test_tensor.py
│   ├── test_broadcasting.py
│   ├── test_conv.py
│   ├── test_xor_model.py
│   └── test_bfgs.py
│
├── LIMITATIONS.md
└── pytest.ini

```

## Running Tests

```bash
pytest -v
```

---

## Motivation

Modern ML frameworks abstract away much of the underlying mathematical and computational machinery. Building nanograd from scratch develops intuition for:

- Reverse-mode automatic differentiation
- Computational graph construction and traversal
- Tensor broadcasting semantics
- Neural network layer abstractions
- Gradient-based optimization — first and second order
- Why quasi-Newton methods converge faster than gradient descent, and what it costs

---

## Limitations

See LIMITATIONS.md for a detailed discussion of known constraints and design tradeoffs.

---

## Future Work

- Adam optimizer
- Forward-mode AD (JVP) via dual numbers
- Exact Hessians via reverse-over-reverse (prototype in `second-order-autodiff` branch)
- Stride and padding support for convolutions
- Batched inference support