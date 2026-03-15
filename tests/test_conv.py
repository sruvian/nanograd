import numpy as np
from tensors import Tensor
from layers.conv import Conv1D, Conv2D

def test_conv1d_simple():
    x = Tensor(np.array([[1., 2., 3., 4.]]))
    conv = Conv1D(kernel_size=2, input_channels=1, output_channels=1)
    conv.kernel_vector.data = np.array([[[1., 1.]]])
    y = conv(x)
    expected = np.array([[3., 5., 7.]])
    assert np.allclose(y.data, expected)

def test_conv1d_shape():
    x = Tensor(np.random.randn(2, 10))
    conv = Conv1D(kernel_size=3, input_channels=2, output_channels=4)
    y = conv(x)
    assert y.shape == (4, 8)

def test_conv1d_backward():
    x = Tensor(np.random.randn(2, 8))
    conv = Conv1D(kernel_size=3, input_channels=2, output_channels=3)
    y = conv(x)
    loss = y.sum()
    loss.backward()
    assert x.g.shape == x.shape
    assert conv.kernel_vector.g.shape == conv.kernel_vector.shape

def test_conv1d_multichannel():
    x = Tensor(np.array([
        [1., 2., 3., 4.],
        [0., 1., 0., 1.]
    ]))
    conv = Conv1D(kernel_size=2, input_channels=2, output_channels=1)
    conv.kernel_vector.data = np.array([[
        [1., 0.],
        [0., 1.]
    ]])
    y = conv(x)
    expected = np.array([[2., 2., 4.]])
    assert np.allclose(y.data, expected)

def test_conv2d_simple():
    x = Tensor(np.array([[
        [1., 2., 3.],
        [4., 5., 6.],
        [7., 8., 9.]
    ]]))
    conv = Conv2D(kernel_size=2, input_channels=1, output_channels=1)
    conv.kernel_vector.data = np.array([[
        [[1., 1.],
         [1., 1.]]
    ]])
    y = conv(x)
    expected = np.array([[
        [12., 16.],
        [24., 28.]
    ]])
    assert np.allclose(y.data, expected)

def test_conv2d_shape():
    x = Tensor(np.random.randn(3, 10, 10))
    conv = Conv2D(kernel_size=3, input_channels=3, output_channels=5)
    y = conv(x)
    assert y.shape == (5, 8, 8)

def test_conv2d_backward():
    x = Tensor(np.random.randn(3, 8, 8))
    conv = Conv2D(kernel_size=3, input_channels=3, output_channels=2)
    y = conv(x)
    loss = y.sum()
    loss.backward()
    assert x.g.shape == x.shape
    assert conv.kernel_vector.g.shape == conv.kernel_vector.shape

def test_conv2d_multi_channel():
    x = Tensor(np.ones((2, 5, 5)))
    conv = Conv2D(kernel_size=3, input_channels=2, output_channels=1)
    y = conv(x)
    assert y.shape == (1, 3, 3)