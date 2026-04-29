import numpy as np

from tensors import Tensor
from layers import Linear, ReLU, Sigmoid, Sequential, Tanh
from losses import MSE
from optimizers import SGD


def test_xor_learning():
    """SGD should solve XOR — tested across multiple seeds for robustness."""
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
        optimizer = SGD(model.parameters(), lr=0.1)

        for _ in range(2000):
            preds = model(X)
            loss = loss_fn(y, preds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        preds = model(X).data
        if preds[0] < 0.3 and preds[1] > 0.7 and preds[2] > 0.7 and preds[3] < 0.3:
            solved = True
            break

    assert solved, "SGD failed to solve XOR across multiple seeds"