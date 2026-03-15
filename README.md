# nanograd

nanograd is a minimal deep learning framework implementing reverse-mode automatic differentiation from scratch in Python.

The project builds a small neural network library on top of a custom autodiff engine, supporting tensor operations, neural network layers, convolutional layers, and gradient-based optimization.

The goal is to understand the core mechanics behind modern ML frameworks by implementing them from first principles.

## Features
**Automatic Differentiation**

- Reverse-mode autodiff (backpropagation)

- Computational graph construction

- Topological backward traversal

- Gradient accumulation for reused nodes

* Tensor Operations

    - Broadcasting support

    - Elementwise arithmetic (+, -, *, /, pow)

    - Matrix multiplication

    - Reshape and transpose

    - Tensor slicing / indexing

    - Reductions (sum, mean, max)

* Neural Network Components

- Linear layers

- Activation functions

    - ReLU

    - Tanh

    - Sigmoid

- Sequential model container

- Loss functions

    - Mean Squared Error (MSE)

    * Convolution Layers

    - Conv1D (sliding window convolution)

    - Conv2D (im2col + matrix multiplication)

* Optimization

    - Stochastic Gradient Descent (SGD)

## Testing

* Comprehensive pytest coverage including:

- tensor operations

- broadcasting rules

- gradient correctness

- convolution layers

- neural network training (XOR example)

## Example

### Training a small neural network:
```python
from tensors import Tensor
from layers import Linear, Sequential, Tanh
from losses import MSE
from optim import SGD

model = Sequential(
    Linear(2,4),
    Tanh(),
    Linear(4,1)
)

criterion = MSE()
optimizer = SGD(model.parameters(), lr=0.1)

x = Tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
y = Tensor([[0.],[1.],[1.],[0.]])

for _ in range(2000):

    pred = model(x)
    loss = criterion(y, pred)

    model.zero_grad()
    loss.backward()

    optimizer.step()

print(pred.data)
```
Project Structure
```
nanograd
│
├─ tensors/
│   └─ tensor.py        # core autodiff engine
│
├─ layers/
│   ├─ module.py
│   ├─ linear.py
│   ├─ conv.py
│   └─ activations.py
│
├─ losses/
│   └─ mse.py
│
├─ optim/
│   └─ sgd.py
│
├─ tests/
│   ├─ test_tensor.py
│   ├─ test_broadcasting.py
│   ├─ test_conv.py
│   └─ test_xor_model.py
│
└─ pytest.ini
```

## Running Tests
```bash
pytest -v
```

## Motivation

Modern ML frameworks abstract away much of the underlying mathematical and computational machinery.

Building nanograd from scratch helps develop intuition for:

* reverse-mode automatic differentiation

* computational graph construction

* tensor broadcasting semantics

* neural network layer abstractions

* gradient-based optimization

## Future Work

Planned extensions include:

* Jacobian computation

* Hessian computation

* quasi-Newton optimization (BFGS)

* additional optimizers (Adam)

* stride and padding support for convolutions

* batching support

## License

MIT License