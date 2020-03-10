"""Test base kernels and compositions."""
import numpy as np
from sklearn.gaussian_process.kernels import RBF as sklearn_RBF
from ml_pipeline.base.kernel import RBF


class TestRBF:
    def test_same_as_sklearn(self):
        lengthscale = 15.
        X = np.random.normal(size=(10, 20))

        sk_rbf = sklearn_RBF(lengthscale)
        rbf = RBF(lengthscale)
        assert np.allclose(sk_rbf(X), rbf(X))


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
