import numpy as np
from tensors import Tensor, concat
from jacobians import vec_jac_prod, hess


def test_jacobian_simple():

    x = Tensor(np.array([2.0, 3.0]))

    y1 = x[0] ** 2
    y2 = x[1] ** 2

    y = concat([
        y1.reshape((1,)),
        y2.reshape((1,))
    ], axis=0)


    J = vec_jac_prod(x, y)
    print(J)

    expected = np.array([
        [4.0, 0.0],
        [0.0, 6.0]
    ])

    assert np.allclose(J, expected)

def test_jacobian_mixed():

    x = Tensor(np.array([2.0, 3.0]))

    y1 = x[0] * x[1]
    y2 = x[0] ** 2

    y = concat([
        y1.reshape((1,)),
        y2.reshape((1,))
    ], axis=0)


    J = vec_jac_prod(x, y)

    expected = np.array([
        [3.0, 2.0],
        [4.0, 0.0]
    ])

    assert np.allclose(J, expected)

def test_hessian_quadratic():

    x = Tensor(np.array([2.0, 3.0]))

    f = x[0] ** 2 + x[1] ** 2

    H = hess(f, x)

    expected = np.array([
        [2.0, 0.0],
        [0.0, 2.0]
    ])

    assert np.allclose(H, expected)

def test_hessian_cross_terms():

    x = Tensor(np.array([2.0, 3.0]))

    f = x[0] * x[1]

    H = hess(f, x)

    expected = np.array([
        [0.0, 1.0],
        [1.0, 0.0]
    ])

    assert np.allclose(H, expected)

def test_hessian_is_symmetric():

    x = Tensor(np.array([1.5, 2.5]))

    f = x[0]**3 + x[0]*x[1] + x[1]**2

    H = hess(f, x)

    assert np.allclose(H, H.T)

def test_jacobian_shape():
    x = Tensor(np.array([1.0, 2.0, 3.0]))
    y1 = x[0] + x[1]
    y2 = x[1] * x[2]
    y = concat([y1.reshape((1,)), y2.reshape((1,))], axis=0)
    J = vec_jac_prod(x, y)
    assert J.shape == (2, 3)