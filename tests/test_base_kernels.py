"""Test base kernels and compositions."""
import pytest
from functools import partial
import numpy as np
from sklearn.gaussian_process.kernels import RBF as sklearn_RBF
from sklearn.gaussian_process.kernels import ConstantKernel as sklearn_C

from sklearn_jax_kernels import RBF, ConstantKernel, NormalizedKernel
from sklearn_jax_kernels import config


class TestRBF:
    @pytest.mark.parametrize("save_memory", [True, False])
    def test_value(self, save_memory):
        config.SAVE_MEMORY = save_memory

        lengthscale = 15.
        X = np.random.normal(size=(10, 20))

        sk_rbf = sklearn_RBF(lengthscale)
        rbf = RBF(lengthscale)
        assert np.allclose(sk_rbf(X), rbf(X))

    @pytest.mark.parametrize("save_memory", [True, False])
    def test_gradient(self, save_memory):
        config.SAVE_MEMORY = save_memory

        lengthscale = 1.
        X = np.random.normal(size=(5, 2))

        sk_rbf = sklearn_RBF(lengthscale)
        _, sk_grad = sk_rbf(X, eval_gradient=True)
        rbf = RBF(lengthscale)
        _, grad = rbf(X, eval_gradient=True)
        assert np.allclose(sk_grad, grad)


class TestNormalizedKernel:
    def test_RBF_value_same(self):
        X = np.random.normal(size=(10, 20))
        kernel = NormalizedKernel(RBF(1.))
        K = kernel(X)

        # Compute the kernel using instance wise formulation
        from jax import vmap
        kernel_fn = partial(kernel.pure_kernel_fn, kernel.theta)
        K_instance_wise = \
            vmap(lambda x: vmap(lambda y: kernel_fn(x, y))(X))(X)

        assert np.allclose(K, K_instance_wise)

    def test_RBF_grad_same_XX(self):
        X = np.random.normal(size=(3, 20))
        kernel = NormalizedKernel(RBF(1.))
        K, K_grad = kernel(X, eval_gradient=True)

        # Compute the kernel using instance wise formulation
        from jax import vmap, grad
        kernel_fn = partial(grad(kernel.pure_kernel_fn), kernel.theta)
        K_grad_instance_wise = \
            vmap(lambda x: vmap(lambda y: kernel_fn(x, y))(X))(X)

        assert np.allclose(K_grad, K_grad_instance_wise)

    def test_RBF_grad_same_XY(self):
        X = np.random.normal(size=(3, 20))
        kernel = NormalizedKernel(RBF(1.))
        K, K_grad = kernel(X, X, eval_gradient=True)

        # Compute the kernel using instance wise formulation
        from jax import vmap, grad
        kernel_fn = partial(grad(kernel.pure_kernel_fn), kernel.theta)
        K_grad_instance_wise = \
            vmap(lambda x: vmap(lambda y: kernel_fn(x, y))(X))(X)

        assert np.allclose(K_grad, K_grad_instance_wise)


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
