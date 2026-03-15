from optimizers import SGD
from tensors import Tensor

def test_optimizer_step():
    x = Tensor(2.0)
    y = x * x
    y.backward()
    opt = SGD([x], lr=0.1)
    old_value = x.data.copy()
    opt.step()
    assert x.data != old_value

def test_optimizer_reduces_loss():
    w = Tensor(5.0)
    opt = SGD([w], lr=0.1)
    for _ in range(20):
        loss = w * w
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert abs(w.data) < 5