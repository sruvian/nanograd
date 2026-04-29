import numpy as np
from tensors import Tensor
from jac import numerical_hessian

def test_hessian_scalar_quadratic():
    """f(x) = x^2, H = [2]"""
    x = Tensor(np.array([3.0]))
    
    def f():
        return (x * x).sum()
    
    H = numerical_hessian(f, x)
    expected = np.array([[2.0]])
    assert np.allclose(H, expected, atol=1e-5), f"Expected {expected}, got {H}"

def test_hessian_diagonal():
    """f(x, y) = x^2 + 3y^2, H = [[2, 0], [0, 6]]"""
    x = Tensor(np.array([1.0, 2.0]))
    
    def f():
        return (x * x * Tensor(np.array([1.0, 3.0]), req_grad=False)).sum()
    
    H = numerical_hessian(f, x)
    expected = np.array([[2.0, 0.0], [0.0, 6.0]])
    assert np.allclose(H, expected, atol=1e-5), f"Expected {expected}, got {H}"

def test_hessian_cross_term():
    """f(x, y) = x^2 + xy + y^2, H = [[2, 1], [1, 2]]"""
    x = Tensor(np.array([1.0, 1.0]))
    
    def f():
        x0 = x[0]
        x1 = x[1]
        return (x0 * x0 + x0 * x1 + x1 * x1).sum()
    
    H = numerical_hessian(f, x)
    expected = np.array([[2.0, 1.0], [1.0, 2.0]])
    assert np.allclose(H, expected, atol=1e-5), f"Expected {expected}, got {H}"

def test_hessian_symmetry():
    """Hessian of any smooth function should be symmetric."""
    x = Tensor(np.array([1.5, -0.5, 2.0]))
    
    def f():
        return (x * x * x).sum()  # f = x^3 + y^3 + z^3, H = diag(6x, 6y, 6z)
    
    H = numerical_hessian(f, x)
    assert np.allclose(H, H.T, atol=1e-5), "Hessian should be symmetric"

def test_hessian_cubic_diagonal():
    """f(x) = x^3, H = [6x] at x=2 gives [12]"""
    x = Tensor(np.array([2.0]))
    
    def f():
        return (x * x * x).sum()
    
    H = numerical_hessian(f, x)
    expected = np.array([[12.0]])
    assert np.allclose(H, expected, atol=1e-5), f"Expected {expected}, got {H}"