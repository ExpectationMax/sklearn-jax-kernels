"""Test base kernels and compositions."""
import numpy as np
from sklearn.gaussian_process.kernels import RBF as sklearn_RBF
from sklearn.gaussian_process.kernels import ConstantKernel as sklearn_C

from sklearn_jax_kernels import RBF, ConstantKernel


class TestRBF:
    def test_value(self):
        lengthscale = 15.
        X = np.random.normal(size=(10, 20))

        sk_rbf = sklearn_RBF(lengthscale)
        rbf = RBF(lengthscale)
        assert np.allclose(sk_rbf(X), rbf(X))

    def test_gradient(self):
        lengthscale = 1.
        X = np.random.normal(size=(5, 2))

        sk_rbf = sklearn_RBF(lengthscale)
        _, sk_grad = sk_rbf(X, eval_gradient=True)
        rbf = RBF(lengthscale)
        _, grad = rbf(X, eval_gradient=True)
        assert np.allclose(sk_grad, grad)


class TestConstant:
    def test_value(self):
        val = 5.
        X = np.random.normal(size=(10, 20))

        k = ConstantKernel(val)
        assert np.allclose(k(X), np.full((10, 10), val))

    def test_gradient(self):
        val = 5.
        X = np.random.normal(size=(10, 20))

        sk_c = sklearn_C(val)
        c = ConstantKernel(val)
        _, sk_grad = sk_c(X, eval_gradient=True)
        _, grad = c(X, eval_gradient=True)

        print(sk_grad[0, :, 0], grad[0, :, 0])
        assert np.allclose(sk_grad, grad)



class TestCompositions:
    def test_sum(self):
        lengthscale1, lengthscale2 = 5., 10.
        X = np.random.normal(size=(10, 20))

        sk_sum = sklearn_RBF(lengthscale1) + sklearn_RBF(lengthscale2)
        ours_sum = RBF(lengthscale1) + RBF(lengthscale2)
        assert np.allclose(sk_sum(X), ours_sum(X))

    def test_product(self):
        lengthscale1, lengthscale2 = 5., 10.
        X = np.random.normal(size=(10, 20))

        sk_prod = sklearn_RBF(lengthscale1) + sklearn_RBF(lengthscale2)
        ours_prod = RBF(lengthscale1) + RBF(lengthscale2)
        assert np.allclose(sk_prod(X), ours_prod(X))

    def test_exponentiation(self):
        lengthscale = 5.
        exponent = 2.
        X = np.random.normal(size=(10, 20))

        sk = sklearn_RBF(lengthscale) ** exponent
        ours = RBF(lengthscale) ** exponent
        assert np.allclose(sk(X), ours(X))
