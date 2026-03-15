from layers import Linear, ReLU, Tanh, Sigmoid, Sequential
from tensors import Tensor
import numpy as np

def test_linear_output_shape():
    layer = Linear(3, 2)
    x = Tensor(np.random.randn(5, 3))
    y = layer(x)
    assert y.data.shape == (5, 2)

def test_linear_backward():
    layer = Linear(3, 2)
    x = Tensor(np.random.randn(4, 3))
    y = layer(x)
    loss = y.sum()
    loss.backward()
    assert layer.W.g.shape == layer.W.data.shape
    assert layer.b.g.shape == layer.b.data.shape

def test_linear_parameters():
    layer = Linear(4, 3)
    params = layer.parameters()
    assert len(params) == 2

def test_linear_g_accumulation():
    layer = Linear(3, 2)
    x = Tensor(np.random.randn(4, 3))
    y = layer(x).sum()
    y.backward()
    y.backward()
    assert np.any(layer.W.g.data != 0)

def test_linear_no_bias():
    layer = Linear(3, 2, bias=False)
    x = Tensor(np.random.randn(5, 3))
    y = layer(x)
    loss = y.sum()
    loss.backward()
    assert layer.W.g is not None

def test_relu_module_forward():
    layer = ReLU()
    x = Tensor(np.array([-1.0, 0.0, 2.0]))
    y = layer(x)
    expected = np.array([0.0, 0.0, 2.0])
    assert np.allclose(y.data, expected)

def test_relu_module_backward():
    layer = ReLU()
    x = Tensor(np.array([-1.0, 2.0, 3.0]))
    y = layer(x).sum()
    y.backward()
    expected = np.array([0.0, 1.0, 1.0])
    assert np.allclose(x.g.data, expected)

def test_tanh_module_forward():
    layer = Tanh()
    x = Tensor(np.array([0.0, 1.0]))
    y = layer(x)
    expected = np.tanh(x.data)
    assert np.allclose(y.data, expected)

def test_tanh_module_backward():
    layer = Tanh()
    x = Tensor(np.array([0.0, 1.0]))
    y = layer(x).sum()
    y.backward()
    expected = 1 - np.tanh(x.data)**2
    assert np.allclose(x.g.data, expected)

def test_sigmoid_module_forward():
    layer = Sigmoid()
    x = Tensor(np.array([0.0, 1.0]))
    y = layer(x)
    expected = 1 / (1 + np.exp(-x.data))
    assert np.allclose(y.data, expected)

def test_sigmoid_module_backward():
    layer = Sigmoid()
    x = Tensor(np.array([0.0, 1.0]))
    y = layer(x).sum()
    y.backward()
    sig = 1 / (1 + np.exp(-x.data))
    expected = sig * (1 - sig)
    assert np.allclose(x.g.data, expected)

def test_activation_no_parameters():
    relu_layer = ReLU()
    params = relu_layer.parameters()
    assert params == []

def test_activation_call_interface():
    layer = Sigmoid()
    x = Tensor(0.0)
    y = layer(x)
    assert isinstance(y, Tensor)

def test_sequential_forward():
    model = Sequential(
        Linear(3, 4),
        ReLU(),
        Linear(4, 2)
    )
    x = Tensor(np.random.randn(5, 3))
    y = model(x)
    assert y.data.shape == (5, 2)

def test_sequential_backward():
    model = Sequential(
        Linear(3, 4),
        ReLU(),
        Linear(4, 1)
    )
    x = Tensor(np.random.randn(6, 3))
    y = model(x)
    loss = y.sum()
    loss.backward()
    for p in model.parameters():
        assert p.g is not None
        assert np.any(p.g.data != 0)

def test_sequential_parameters():
    model = Sequential(
        Linear(3, 4),
        Linear(4, 2)
    )
    params = model.parameters()
    assert len(params) == 4

def test_sequential_zero_grad():
    model = Sequential(
        Linear(3, 4),
        Linear(4, 1)
    )
    x = Tensor(np.random.randn(5, 3))
    y = model(x).sum()
    y.backward()
    model.zero_grad()
    for p in model.parameters():
        assert p.grad is None or np.allclose(p.grad.data, 0)

def test_sequential_with_activations():
    model = Sequential(
        Linear(3, 8),
        ReLU(),
        Tanh(),
        Sigmoid(),
        Linear(8, 1)
    )
    x = Tensor(np.random.randn(4, 3))
    y = model(x)
    assert y.data.shape == (4, 1)

def test_sequential_empty():
    model = Sequential()
    x = Tensor(np.random.randn(3, 4))
    y = model(x)
    assert np.allclose(y.data, x.data)