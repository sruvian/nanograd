from layers import Linear
from tensors import Tensor
import numpy as np


def test_linear_output_shape():

    layer = Linear(3,2)

    x = Tensor(np.random.randn(5,3))

    y = layer(x)

    assert y.data.shape == (5,2)


def test_linear_backward():

    layer = Linear(3,2)

    x = Tensor(np.random.randn(4,3))

    y = layer(x)

    loss = y.sum()

    loss.backward()

    assert layer.W.grad.shape == layer.W.data.shape
    assert layer.b.grad.shape == layer.b.data.shape


def test_linear_parameters():

    layer = Linear(4,3)

    params = layer.parameters()

    assert len(params) == 2


def test_linear_grad_accumulation():

    layer = Linear(3,2)

    x = Tensor(np.random.randn(4,3))

    y = layer(x).sum()

    y.backward()
    y.backward()

    assert np.all(layer.W.grad != 0)


def test_linear_no_bias():

    layer = Linear(3,2, bias=False)

    x = Tensor(np.random.randn(5,3))

    y = layer(x)

    loss = y.sum()

    loss.backward()

    assert layer.W.grad is not None