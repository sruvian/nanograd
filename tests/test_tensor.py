from tensors import Tensor, relu, log, exp, sin, sigmoid
import numpy as np

def test_shared_g():
    a = Tensor(2.0)
    b = Tensor(3.0)
    c = Tensor(4.0)
    L = a*b + a*c
    L.backward()
    assert np.allclose(a.g.data, 7.0)
    assert np.allclose(b.g.data, 2.0)
    assert np.allclose(c.g.data, 2.0)

def test_pow_g():
    x = Tensor(5.0)
    y = Tensor(2.0)
    z = (x**3) * (y**2)
    z.backward()
    assert abs(x.g.data - (3 * (x.data**2) * (y.data**2))) < 1e-6
    assert abs(y.g.data - (2 * (x.data**3) * y.data)) < 1e-6

def test_relu():
    x = Tensor(-1.0)
    y = relu(x)
    y.backward()
    assert np.allclose(x.g.data, 0.0)

    x = Tensor(2.0)
    y = relu(x)
    y.backward()
    assert np.allclose(x.g.data, 1.0)

def test_trig():
    x = Tensor(0.0)
    y = sin(x)
    y.backward()
    assert abs(x.g.data - 1.0) < 1e-6

def test_log_exp():
    x = Tensor(2.0)
    y = log(x)
    y.backward()
    assert abs(x.g.data - 0.5) < 1e-6

    x = Tensor(1.0)
    y = exp(x)
    y.backward()
    assert abs(x.g.data - np.exp(1.0)) < 1e-6

def test_scalar_tensor_ops():
    x = Tensor(3.0)
    y = 2 + x
    y.backward()
    assert np.allclose(y.data, 5.0)
    assert np.allclose(x.g.data, 1.0)

    x = Tensor(3.0)  # fresh tensor instead of mutating g
    y = 2 * x
    y.backward()
    assert np.allclose(y.data, 6.0)
    assert np.allclose(x.g.data, 2.0)

def test_reuse_node_multiple_times():
    x = Tensor(2.0)
    y = x * x + x
    y.backward()
    assert np.allclose(x.g.data, 5.0)

def test_division():
    x = Tensor(6.0)
    y = Tensor(3.0)
    z = x / y
    z.backward()
    assert abs(x.g.data - (1/3)) < 1e-6
    assert abs(y.g.data - (-6/9)) < 1e-6

def test_backward_twice_accumulates():
    # g accumulates across backward calls — this is expected behavior
    x = Tensor(2.0)
    y = x * 3
    y.backward()
    y.backward()
    assert np.allclose(x.g.data, 6.0)  # 3 + 3 accumulated

def test_requires_grad():
    x = Tensor(2.0, req_grad=False)
    y = Tensor(3.0)
    z = x * y
    z.backward()
    assert x.grad is None
    assert np.allclose(y.grad.data, 2.0)
    
def test_sigmoid_forward():
    x = Tensor(0.0)
    y = sigmoid(x)
    assert abs(y.data - 0.5) < 1e-6

def test_sigmoid_backward():
    x = Tensor(0.0)
    y = sigmoid(x)
    y.backward()
    assert abs(x.g.data - 0.25) < 1e-6

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
    assert np.allclose(x.g.data, expected)

def test_sigmoid_chain_rule():
    x = Tensor(1.0)
    y = sigmoid(x) * 2
    y.backward()
    sig = 1 / (1 + np.exp(-1.0))
    expected = 2 * sig * (1 - sig)
    assert abs(x.g.data - expected) < 1e-6