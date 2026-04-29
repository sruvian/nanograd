import numpy as np
from tensors import Tensor
from layers import Linear, Tanh, Sigmoid, Sequential
from losses import MSE
from optimizers import BFGS

def test_bfgs_step_updates_params():
    """BFGS should update parameters after one step."""
    w = Tensor(np.array([3.0, -2.0]))
    old_data = w.data.copy()

    def closure():
        loss = (w * w).sum()
        loss.backward()
        return float(loss.data)

    opt = BFGS([w])
    opt.step(closure)

    assert not np.allclose(w.data, old_data), "Parameters should have changed"

def test_bfgs_reduces_quadratic():
    """BFGS should minimise a simple quadratic in very few steps."""
    w = Tensor(np.array([5.0, -4.0, 3.0]))
    opt = BFGS([w])

    for _ in range(10):
        opt.zero_grad()
        def closure():
            opt.zero_grad()
            loss = (w * w).sum()
            loss.backward()
            return float(loss.data)
        opt.step(closure)

    assert float((w * w).sum().data) < 1e-6, "Should converge to near zero"

def test_bfgs_xor():
    """BFGS should solve XOR — tested across multiple seeds for robustness."""
    X = Tensor(np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ]))
    y = Tensor(np.array([
        [0.0],
        [1.0],
        [1.0],
        [0.0]
    ]))

    seeds = [0, 1, 42, 123, 7]
    solved = False

    for seed in seeds:
        np.random.seed(seed)
        model = Sequential(
            Linear(2, 4),
            Tanh(),
            Linear(4, 1),
            Sigmoid()
        )
        loss_fn = MSE("mean")
        opt = BFGS(model.parameters())

        def closure():
            opt.zero_grad()
            preds = model(X)
            loss = loss_fn(y, preds)
            loss.backward()
            return float(loss.data)

        for _ in range(200):
            opt.step(closure)

        preds = model(X).data
        if preds[0] < 0.3 and preds[1] > 0.7 and preds[2] > 0.7 and preds[3] < 0.3:
            solved = True
            break

    assert solved, "BFGS failed to solve XOR across multiple seeds"

def test_bfgs_faster_than_sgd():
    """BFGS should reach low loss in fewer steps than SGD on a quadratic."""
    from optimizers import SGD

    # BFGS
    w_bfgs = Tensor(np.array([5.0, -4.0, 3.0]))
    opt_bfgs = BFGS([w_bfgs])
    bfgs_steps = 0

    for _ in range(50):
        def closure():
            opt_bfgs.zero_grad()
            loss = (w_bfgs * w_bfgs).sum()
            loss.backward()
            return float(loss.data)
        opt_bfgs.step(closure)
        bfgs_steps += 1
        if float((w_bfgs * w_bfgs).sum().data) < 1e-6:
            break

    # SGD
    w_sgd = Tensor(np.array([5.0, -4.0, 3.0]))
    opt_sgd = SGD([w_sgd], lr=0.1)
    sgd_steps = 0

    for _ in range(500):
        opt_sgd.zero_grad()
        loss = (w_sgd * w_sgd).sum()
        loss.backward()
        opt_sgd.step()
        sgd_steps += 1
        if float((w_sgd * w_sgd).sum().data) < 1e-6:
            break

    assert bfgs_steps < sgd_steps, \
        f"BFGS took {bfgs_steps} steps, SGD took {sgd_steps} — BFGS should be faster"