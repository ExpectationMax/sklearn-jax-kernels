
import pytest
import numpy as np
from inspect import signature

from sklearn.gaussian_process.kernels import _approx_fprime
from sklearn.base import clone
from sklearn.utils._testing import (assert_almost_equal, assert_array_equal,
                                    assert_array_almost_equal)

from sklearn_jax_kernels import (
    RBF, Kernel, KernelOperator, ConstantKernel, Exponentiation)

from jax.config import config
config.update("jax_enable_x64", True)  # Required for numerical gradients checks

X = np.random.RandomState(0).normal(0, 1, (5, 2))
Y = np.random.RandomState(0).normal(0, 1, (6, 2))

kernels = [RBF(length_scale=2.0), RBF(length_scale_bounds=(0.5, 2.0)),
           ConstantKernel(constant_value=10.0),
           2.0 * RBF(length_scale=0.33, length_scale_bounds="fixed"),
           2.0 * RBF(length_scale=0.5),
           2.0 * RBF(length_scale=[0.5, 2.0]),
           RBF(length_scale=[2.0])]


@pytest.mark.parametrize('kernel', kernels)
def test_kernel_gradient(kernel):
    # Compare analytic and numeric gradient of kernels.
    K, K_gradient = kernel(X, eval_gradient=True)

    assert K_gradient.shape[0] == X.shape[0]
    assert K_gradient.shape[1] == X.shape[0]
    assert K_gradient.shape[2] == kernel.theta.shape[0]

    def eval_kernel_for_theta(theta):
        kernel_clone = kernel.clone_with_theta(theta)
        K = kernel_clone(X, eval_gradient=False)
        return K

    K_gradient_approx = \
        _approx_fprime(kernel.theta, eval_kernel_for_theta, 1e-10)

    assert_almost_equal(K_gradient, K_gradient_approx, 5)


@pytest.mark.parametrize(
        'kernel',
        [kernel for kernel in kernels
         # skip non-basic kernels
         if not (isinstance(kernel, KernelOperator)
                 or isinstance(kernel, Exponentiation))])
def test_kernel_theta(kernel):
    # Check that parameter vector theta of kernel is set correctly.
    theta = kernel.theta
    _, K_gradient = kernel(X, eval_gradient=True)

    # Determine kernel parameters that contribute to theta
    init_sign = signature(kernel.__class__.__init__).parameters.values()
    args = [p.name for p in init_sign if p.name != 'self']
    theta_vars = map(lambda s: s[0:-len("_bounds")],
                     filter(lambda s: s.endswith("_bounds"), args))
    assert (
        set(hyperparameter.name
            for hyperparameter in kernel.hyperparameters) ==
        set(theta_vars))

    # Check that values returned in theta are consistent with
    # hyperparameter values (being their logarithms)
    for i, hyperparameter in enumerate(kernel.hyperparameters):
        assert (theta[i] == np.log(getattr(kernel, hyperparameter.name)))

    # Fixed kernel parameters must be excluded from theta and gradient.
    for i, hyperparameter in enumerate(kernel.hyperparameters):
        # create copy with certain hyperparameter fixed
        params = kernel.get_params()
        params[hyperparameter.name + "_bounds"] = "fixed"
        kernel_class = kernel.__class__
        new_kernel = kernel_class(**params)
        # Check that theta and K_gradient are identical with the fixed
        # dimension left out
        _, K_gradient_new = new_kernel(X, eval_gradient=True)
        assert theta.shape[0] == new_kernel.theta.shape[0] + 1
        assert K_gradient.shape[2] == K_gradient_new.shape[2] + 1
        if i > 0:
            assert theta[:i] == new_kernel.theta[:i]
            assert_array_equal(K_gradient[..., :i],
                               K_gradient_new[..., :i])
        if i + 1 < len(kernel.hyperparameters):
            assert theta[i + 1:] == new_kernel.theta[i:]
            assert_array_equal(K_gradient[..., i + 1:],
                               K_gradient_new[..., i:])

    # Check that values of theta are modified correctly
    for i, hyperparameter in enumerate(kernel.hyperparameters):
        theta[i] = np.log(42)
        kernel.theta = theta
        assert_almost_equal(getattr(kernel, hyperparameter.name), 42)

        setattr(kernel, hyperparameter.name, 43)
        assert_almost_equal(kernel.theta[i], np.log(43))


@pytest.mark.parametrize('kernel', kernels)
def test_auto_vs_cross(kernel):
    # Auto-correlation and cross-correlation should be consistent.
    K_auto = kernel(X)
    K_cross = kernel(X, X)
    print(K_auto, K_cross)
    assert_array_almost_equal(K_auto, K_cross, 5)


@pytest.mark.parametrize('kernel', kernels)
def test_kernel_diag(kernel):
    # Test that diag method of kernel returns consistent results.
    K_call_diag = np.diag(kernel(X))
    K_diag = kernel.diag(X)
    assert_array_almost_equal(K_call_diag, K_diag, 5)


def test_kernel_operator_commutative():
    # Adding kernels and multiplying kernels should be commutative.
    # Check addition
    assert_array_almost_equal((RBF(2.0) + 1.0)(X), (1.0 + RBF(2.0))(X))

    # Check multiplication
    assert_array_almost_equal((3.0 * RBF(2.0))(X), (RBF(2.0) * 3.0)(X))


def test_kernel_anisotropic():
    # Anisotropic kernel should be consistent with isotropic kernels.
    kernel = 3.0 * RBF([0.5, 2.0])

    K = kernel(X)
    X1 = np.array(X)
    X1[:, 0] *= 4
    K1 = 3.0 * RBF(2.0)(X1)
    assert_array_almost_equal(K, K1)

    X2 = np.array(X)
    X2[:, 1] /= 4
    K2 = 3.0 * RBF(0.5)(X2)
    assert_array_almost_equal(K, K2)

    # Check getting and setting via theta
    kernel.theta = kernel.theta + np.log(2)
    assert_array_equal(kernel.theta, np.log([6.0, 1.0, 4.0]))
    assert_array_equal(kernel.k2.length_scale, [1.0, 4.0])


@pytest.mark.parametrize('kernel',
                         [kernel for kernel in kernels
                          if kernel.is_stationary()])
def test_kernel_stationary(kernel):
    # Test stationarity of kernels.
    K = kernel(X, X + 1)
    assert_almost_equal(K[0, 0], np.diag(K))


@pytest.mark.parametrize('kernel',  kernels)
def test_kernel_input_type(kernel):
    # Test whether kernels is for vectors or structured data
    if isinstance(kernel, Exponentiation):
        assert(kernel.requires_vector_input ==
               kernel.kernel.requires_vector_input)
    if isinstance(kernel, KernelOperator):
        assert(kernel.requires_vector_input ==
               (kernel.k1.requires_vector_input or
                kernel.k2.requires_vector_input))


def check_hyperparameters_equal(kernel1, kernel2):
    # Check that hyperparameters of two kernels are equal
    for attr in set(dir(kernel1) + dir(kernel2)):
        if attr.startswith("hyperparameter_"):
            attr_value1 = getattr(kernel1, attr)
            attr_value2 = getattr(kernel2, attr)
            assert attr_value1 == attr_value2


@pytest.mark.parametrize("kernel", kernels)
def test_kernel_clone(kernel):
    # Test that sklearn's clone works correctly on kernels.
    kernel_cloned = clone(kernel)

    # XXX: Should this be fixed?
    # This differs from the sklearn's estimators equality check.
    assert kernel == kernel_cloned
    assert id(kernel) != id(kernel_cloned)

    # Check that all constructor parameters are equal.
    assert kernel.get_params() == kernel_cloned.get_params()

    # Check that all hyperparameters are equal.
    check_hyperparameters_equal(kernel, kernel_cloned)


@pytest.mark.parametrize('kernel', kernels)
def test_kernel_clone_after_set_params(kernel):
    # This test is to verify that using set_params does not
    # break clone on kernels.
    # This used to break because in kernels such as the RBF, non-trivial
    # logic that modified the length scale used to be in the constructor
    # See https://github.com/scikit-learn/scikit-learn/issues/6961
    # for more details.
    bounds = (1e-5, 1e5)
    kernel_cloned = clone(kernel)
    params = kernel.get_params()
    # RationalQuadratic kernel is isotropic.
    isotropic_kernels = ()
    if 'length_scale' in params and not isinstance(kernel,
                                                   isotropic_kernels):
        length_scale = params['length_scale']
        if np.iterable(length_scale):
            # XXX unreached code as of v0.22
            params['length_scale'] = length_scale[0]
            params['length_scale_bounds'] = bounds
        else:
            params['length_scale'] = [length_scale] * 2
            params['length_scale_bounds'] = bounds * 2
        kernel_cloned.set_params(**params)
        kernel_cloned_clone = clone(kernel_cloned)
        assert (kernel_cloned_clone.get_params() == kernel_cloned.get_params())
        assert id(kernel_cloned_clone) != id(kernel_cloned)
        check_hyperparameters_equal(kernel_cloned, kernel_cloned_clone)


@pytest.mark.parametrize("kernel", kernels)
def test_set_get_params(kernel):
    # Check that set_params()/get_params() is consistent with kernel.theta.

    # Test get_params()
    index = 0
    params = kernel.get_params()
    for hyperparameter in kernel.hyperparameters:
        if isinstance("string", type(hyperparameter.bounds)):
            if hyperparameter.bounds == "fixed":
                continue
        size = hyperparameter.n_elements
        if size > 1:  # anisotropic kernels
            assert_almost_equal(np.exp(kernel.theta[index:index + size]),
                                params[hyperparameter.name])
            index += size
        else:
            assert_almost_equal(np.exp(kernel.theta[index]),
                                params[hyperparameter.name])
            index += 1
    # Test set_params()
    index = 0
    value = 10  # arbitrary value
    for hyperparameter in kernel.hyperparameters:
        if isinstance("string", type(hyperparameter.bounds)):
            if hyperparameter.bounds == "fixed":
                continue
        size = hyperparameter.n_elements
        if size > 1:  # anisotropic kernels
            kernel.set_params(**{hyperparameter.name: [value] * size})
            assert_almost_equal(np.exp(kernel.theta[index:index + size]),
                                [value] * size)
            index += size
        else:
            kernel.set_params(**{hyperparameter.name: value})
            assert_almost_equal(np.exp(kernel.theta[index]), value)
            index += 1


@pytest.mark.parametrize("kernel", kernels)
def test_repr_kernels(kernel):
    # Smoke-test for repr in kernels.

    repr(kernel)


def test_warns_on_get_params_non_attribute():
    class MyKernel(Kernel):
        def __init__(self, param=5):
            pass

        def __call__(self, X, Y=None, eval_gradient=False):
            return X

        def diag(self, X):
            return np.ones(X.shape[0])

        def is_stationary(self):
            return False

        @property
        def pure_kernel_fn(self):
            pass

    est = MyKernel()
    with pytest.warns(FutureWarning, match='AttributeError'):
        params = est.get_params()

    assert params['param'] is None
