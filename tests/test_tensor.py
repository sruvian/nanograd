from tensors import Tensor
from tensors import relu, log, exp, sin, sigmoid
import numpy as np


def test_shared_grad():
    a = Tensor(2.0)
    b = Tensor(3.0)
    c = Tensor(4.0)

    L = a*b + a*c
    L.backward()

    assert a.grad == 7
    assert b.grad == 2
    assert c.grad == 2


def test_pow_grad():
    x = Tensor(5.0)
    y = Tensor(2.0)

    z = (x**3) * (y**2)
    z.backward()

    assert abs(x.grad - (3 * (x.data**2) * (y.data**2))) < 1e-6
    assert abs(y.grad - (2 * (x.data**3) * y.data)) < 1e-6


def test_relu():
    x = Tensor(-1.0)
    y = relu(x)
    y.backward()
    assert x.grad == 0.0

    x = Tensor(2.0)
    y = relu(x)
    y.backward()
    assert x.grad == 1.0


def test_trig():
    x = Tensor(0.0)
    y = sin(x)
    y.backward()
    assert abs(x.grad - 1.0) < 1e-6


def test_log_exp():
    x = Tensor(2.0)
    y = log(x)
    y.backward()
    assert abs(x.grad - 0.5) < 1e-6

    x = Tensor(1.0)
    y = exp(x)
    y.backward()
    assert abs(x.grad - np.exp(1.0)) < 1e-6


def test_scalar_tensor_ops():
    x = Tensor(3.0)

    y = 2 + x
    y.backward()
    assert y.data == 5.0
    assert x.grad == 1.0

    x.grad = np.array(0.0)
    y = 2 * x
    y.backward()
    assert y.data == 6.0
    assert x.grad == 2.0


def test_reuse_node_multiple_times():
    x = Tensor(2.0)

    y = x * x + x
    y.backward()

    assert x.grad == 5.0 


def test_division():
    x = Tensor(6.0)
    y = Tensor(3.0)

    z = x / y
    z.backward()

    assert abs(x.grad - (1 / 3)) < 1e-6
    assert abs(y.grad - (-6 / 9)) < 1e-6


def test_backward_twice_accumulates():
    x = Tensor(2.0)

    y = x * 3
    y.backward()
    y.backward()

    assert x.grad == 6.0

def test_requires_grad():
    x = Tensor(2.0, req_grad=False)
    y = Tensor(3.0)

    z = x * y
    z.backward()

    assert x.grad == 0.0
    assert y.grad == 2.0


def test_sigmoid_forward():

    x = Tensor(0.0)

    y = sigmoid(x)

    assert abs(y.data - 0.5) < 1e-6

def test_sigmoid_backward():

    x = Tensor(0.0)

    y = sigmoid(x)

    y.backward()

    expected = 0.25

    assert abs(x.grad - expected) < 1e-6


def test_sigmoid_vector():

    x = Tensor(np.array([0.0, 1.0, -1.0]))

    y = sigmoid(x)

    expected = 1 / (1 + np.exp(-x.data))

    assert np.allclose(y.data, expected)


def test_sigmoid_vector_backward():

    x = Tensor(np.array([0.0, 1.0, -1.0]))

    y = sigmoid(x).sum()

    y.backward()

    sig = 1 / (1 + np.exp(-x.data))

    expected = sig * (1 - sig)

    assert np.allclose(x.grad, expected)

def test_sigmoid_chain_rule():

    x = Tensor(1.0)

    y = sigmoid(x) * 2

    y.backward()

    sig = 1 / (1 + np.exp(-1))

    expected = 2 * sig * (1 - sig)

    assert abs(x.grad - expected) < 1e-6