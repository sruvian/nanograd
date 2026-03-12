from tensors import Tensor
import numpy as np

def test_vector_add():
    x = Tensor(np.array([1.0, 2.0, 3.0]))
    y = Tensor(np.array([4.0, 5.0, 6.0]))

    z = x + y
    loss = z.sum()
    loss.backward()

    assert np.allclose(x.grad, np.ones(3))
    assert np.allclose(y.grad, np.ones(3))

def test_scalar_broadcast_vector():
    x = Tensor(np.array([1.0, 2.0, 3.0]))
    y = Tensor(2.0)

    z = x + y
    loss = z.sum()
    loss.backward()

    assert np.allclose(x.grad, np.ones(3))
    assert np.allclose(y.grad, 3.0)

def test_vector_matrix_broadcast():
    x = Tensor(np.ones((3, 4)))
    y = Tensor(np.ones(4))

    z = x + y
    loss = z.sum()
    loss.backward()

    assert np.allclose(x.grad, np.ones((3,4)))
    assert np.allclose(y.grad, np.full(4, 3))

def test_column_broadcast():
    x = Tensor(np.ones((3,4)))
    y = Tensor(np.ones((3,1)))

    z = x + y
    loss = z.sum()
    loss.backward()

    assert np.allclose(x.grad, np.ones((3,4)))
    assert np.allclose(y.grad, np.full((3,1), 4))

def test_row_broadcast():
    x = Tensor(np.ones((3,4)))
    y = Tensor(np.ones((1,4)))

    z = x + y
    loss = z.sum()
    loss.backward()

    assert np.allclose(y.grad, np.full((1,4), 3))

def test_double_broadcast():
    x = Tensor(np.ones((3,1)))
    y = Tensor(np.ones((1,4)))

    z = x + y
    loss = z.sum()
    loss.backward()

    assert np.allclose(x.grad, np.full((3,1), 4))
    assert np.allclose(y.grad, np.full((1,4), 3))

def test_broadcast_reuse():
    x = Tensor(np.ones((3,4)))
    y = Tensor(np.ones(4))

    z = x + y + y
    loss = z.sum()
    loss.backward()

    assert np.allclose(y.grad, np.full(4, 6))

def test_mul_broadcast():
    x = Tensor(np.ones((3,4)))
    y = Tensor(np.ones(4) * 2)

    z = x * y
    loss = z.sum()
    loss.backward()

    assert np.allclose(x.grad, np.full((3,4), 2))
    assert np.allclose(y.grad, np.full(4, 3))

def test_vector_dot():
    x = Tensor(np.array([1.0,2.0,3.0]))
    y = Tensor(np.array([4.0,5.0,6.0]))

    z = (x * y).sum()
    z.backward()

    assert np.allclose(x.grad, y.data)
    assert np.allclose(y.grad, x.data)

def test_matmul_forward():
    A = Tensor(np.array([[1.,2.],[3.,4.]]))
    B = Tensor(np.array([[5.,6.],[7.,8.]]))

    C = A @ B

    expected = np.array([[19.,22.],[43.,50.]])
    assert np.allclose(C.data, expected)

def test_matmul_backward():
    A = Tensor(np.random.randn(3,4))
    B = Tensor(np.random.randn(4,2))

    C = A @ B
    loss = C.sum()
    loss.backward()

    assert np.allclose(A.grad, np.ones((3,2)) @ B.data.T)
    assert np.allclose(B.grad, A.data.T @ np.ones((3,2)))

def test_matmul_chain():
    A = Tensor(np.random.randn(3,4))
    B = Tensor(np.random.randn(4,5))
    C = Tensor(np.random.randn(5,2))

    out = (A @ B) @ C
    loss = out.sum()
    loss.backward()

    assert A.grad.shape == A.data.shape
    assert B.grad.shape == B.data.shape
    assert C.grad.shape == C.data.shape

def test_linear_layer_pattern():
    X = Tensor(np.random.randn(5,3))
    W = Tensor(np.random.randn(3,4))
    b = Tensor(np.random.randn(4))

    Y = X @ W + b
    loss = Y.sum()
    loss.backward()

    assert W.grad.shape == W.data.shape
    assert b.grad.shape == b.data.shape

def test_high_dim_broadcast():
    x = Tensor(np.ones((2,3,4)))
    y = Tensor(np.ones((1,3,4)))

    z = x + y
    loss = z.sum()
    loss.backward()

    assert np.allclose(y.grad, np.full((1,3,4), 2))


def test_multi_broadcast_grad():
    x = Tensor(np.ones((3,4)))
    y = Tensor(np.ones((4)))

    z = x * y + x * y
    loss = z.sum()
    loss.backward()

    assert np.allclose(y.grad, np.full(4, 6))


def test_grad_shape_matches_data():
    x = Tensor(np.random.randn(3,4))
    y = (x * 2).sum()
    y.backward()

    assert x.grad.shape == x.data.shape


def test_sum_backward():
    x = Tensor(np.ones((3,4)))
    y = x.sum()
    y.backward()

    assert np.allclose(x.grad, np.ones((3,4)))


def test_reshape_backward():
    x = Tensor(np.arange(6).reshape(2,3).astype(float))

    y = x.reshape((3,2))
    loss = y.sum()

    loss.backward()

    assert np.allclose(x.grad, np.ones((2,3)))


def test_transpose_backward():
    x = Tensor(np.random.randn(3,4))

    y = x.T
    loss = y.sum()

    loss.backward()

    assert np.allclose(x.grad, np.ones((3,4)))