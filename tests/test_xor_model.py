import numpy as np

from tensors import Tensor
from layers import Linear, ReLU, Sigmoid, Sequential, Tanh
from losses import MSE
from optimizers import SGD


def test_xor_learning():

    # XOR dataset
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

    # model
    model = Sequential(
        Linear(2, 4),
        Tanh(),
        Linear(4, 1),
        Sigmoid()
    )

    loss_fn = MSE("mean")

    optimizer = SGD(model.parameters(), lr=0.1)

    # training loop
    for _ in range(2000):

        preds = model(X)

        loss = loss_fn(y, preds)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    preds = model(X).data

    # XOR predictions should be close
    assert preds[0] < 0.3
    assert preds[1] > 0.7
    assert preds[2] > 0.7
    assert preds[3] < 0.3