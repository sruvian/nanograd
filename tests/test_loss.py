from losses import MSE
from tensors import Tensor
import numpy as np
import pytest

def test_mse_mean():
    y_true = Tensor([1.0, 2.0, 3.0])
    y_pred = Tensor([1.0, 2.0, 4.0])
    loss_fn = MSE("mean")
    loss = loss_fn(y_true, y_pred)
    expected = ((y_true.data - y_pred.data) ** 2).mean()
    assert abs(loss.data - expected) < 1e-6

def test_mse_sum():
    y_true = Tensor([1.0, 2.0, 3.0])
    y_pred = Tensor([1.0, 2.0, 4.0])
    loss_fn = MSE("sum")
    loss = loss_fn(y_true, y_pred)
    expected = ((y_true.data - y_pred.data) ** 2).sum()
    assert abs(loss.data - expected) < 1e-6

def test_mse_backward():
    y_true = Tensor([1.0, 2.0, 3.0])
    y_pred = Tensor([1.0, 2.0, 4.0])
    loss_fn = MSE("mean")
    loss = loss_fn(y_true, y_pred)
    loss.backward()
    expected = 2 * (y_pred.data - y_true.data) / 3
    assert np.allclose(y_pred.g.data, expected)

def test_invalid_reduction():
    with pytest.raises(ValueError):
        MSE("invalid")

def test_mse_returns_tensor():
    y_true = Tensor([1.0, 2.0])
    y_pred = Tensor([1.0, 3.0])
    loss_fn = MSE()
    loss = loss_fn(y_true, y_pred)
    assert isinstance(loss, Tensor)