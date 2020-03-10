"""Base class of Kernel implementation compatible with JAX."""
import abc
from functools import partial
from sklearn.gaussian_process.kernels import Kernel as sklearn_kernel
from sklearn.gaussian_process.kernels import (
    Hyperparameter,
    StationaryKernelMixin,
    NormalizedKernelMixin
)
from jax import jit, vmap, value_and_grad
import jax.numpy as np


class Kernel(sklearn_kernel, metaclass=abc.ABCMeta):
    """Kernel object similar to sklearn implementation but supporting JAX.

    Contains additional methods:
     - pure_kernel_function

    """

    @property
    @abc.abstractmethod
    def pure_kernel_fn(self):
        """Get a pure function which applies the kernel.

        Returned function should have the signature:
        fn(theta, x1, x2).
        """

    def get_kernel_matrix_fn(self, eval_gradient):
        """Return pure function for computing kernel matrix and gradients.

        Returned function has the signature: `f(theta, X, Y)`
        """
        pure_kernel_fn = self.pure_kernel_fn

        def kernel_matrix_fn(theta, X, Y):
            kernel_fn = pure_kernel_fn
            if eval_gradient:
                kernel_fn = value_and_grad(pure_kernel_fn)
            kernel_fn = jit(partial(kernel_fn, theta))
            return vmap(lambda x: vmap(lambda y: kernel_fn(x, y))(Y))(X)

        return kernel_matrix_fn

    def __call__(self, X, Y=None, eval_gradient=False):
        """Build kernel matrix from input data X and Y.

        Evtl. also compute the gradient with respect to the parameters.
        """
        if Y is None:
            Y = X

        return self.get_kernel_matrix_fn(eval_gradient)(self.theta, X, Y)

    def diag(self, X):
        """Get diagonal of kernel matrix."""
        return vmap(lambda x: self.pure_kernel_fn(self.theta, x, x))(X)

    def __add__(self, b):
        """Add kernel to constant or other kernel."""
        if not isinstance(b, Kernel):
            return Sum(self, ConstantKernel(b))
        return Sum(self, b)

    def __radd__(self, b):
        """Add kernel to constant or other kernel."""
        if not isinstance(b, Kernel):
            return Sum(ConstantKernel(b), self)
        return Sum(b, self)

    def __mul__(self, b):
        """Mulitply kernel with constant or other kernel."""
        if not isinstance(b, Kernel):
            return Product(self, ConstantKernel(b))
        return Product(self, b)

    def __rmul__(self, b):
        """Mulitply kernel with constant or other kernel."""
        if not isinstance(b, Kernel):
            return Product(ConstantKernel(b), self)
        return Product(b, self)

    def __pow__(self, b):
        """Exponentiate kernel."""
        return Exponentiation(self, b)


class KernelOperator(Kernel):
    """Base class for all kernel operators."""

    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    def get_params(self, deep=True):
        """Get parameters of this kernel.

        Parameters:
            deep : boolean, optional
                If True, will return the parameters for this estimator and
                contained subobjects that are estimators.

        Returns:
            params : mapping of string to any
                Parameter names mapped to their values.

        """
        params = dict(k1=self.k1, k2=self.k2)
        if deep:
            deep_items = self.k1.get_params().items()
            params.update(('k1__' + k, val) for k, val in deep_items)
            deep_items = self.k2.get_params().items()
            params.update(('k2__' + k, val) for k, val in deep_items)

        return params

    @property
    def hyperparameters(self):
        """Return a list of all hyperparameter."""
        r = [Hyperparameter("k1__" + hyperparameter.name,
                            hyperparameter.value_type,
                            hyperparameter.bounds, hyperparameter.n_elements)
             for hyperparameter in self.k1.hyperparameters]

        for hyperparameter in self.k2.hyperparameters:
            r.append(Hyperparameter("k2__" + hyperparameter.name,
                                    hyperparameter.value_type,
                                    hyperparameter.bounds,
                                    hyperparameter.n_elements))
        return r

    @property
    def theta(self):
        """Return the (flattened, log-transformed) non-fixed hyperparameters.

        Note that theta are typically the log-transformed values of the
        kernel's hyperparameters as this representation of the search space
        is more amenable for hyperparameter search, as hyperparameters like
        length-scales naturally live on a log-scale.

        Returns:
            theta : array, shape (n_dims,)
                The non-fixed, log-transformed hyperparameters of the kernel

        """
        return np.append(self.k1.theta, self.k2.theta)

    @theta.setter
    def theta(self, theta):
        """Set the (flattened, log-transformed) non-fixed hyperparameters.

        Parameters:
            theta : array, shape (n_dims,)
                The non-fixed, log-transformed hyperparameters of the kernel

        """
        k1_dims = self.k1.n_dims
        self.k1.theta = theta[:k1_dims]
        self.k2.theta = theta[k1_dims:]

    @property
    def bounds(self):
        """Return the log-transformed bounds on the theta.

        Returns:
            bounds : array, shape (n_dims, 2)
                The log-transformed bounds on the kernel's hyperparameters
                theta

        """
        if self.k1.bounds.size == 0:
            return self.k2.bounds
        if self.k2.bounds.size == 0:
            return self.k1.bounds
        return np.vstack((self.k1.bounds, self.k2.bounds))

    def __eq__(self, b):
        """Check for equality between kernels."""
        if type(self) != type(b):
            return False
        return (self.k1 == b.k1 and self.k2 == b.k2) \
            or (self.k1 == b.k2 and self.k2 == b.k1)

    def is_stationary(self):
        """Return whether the kernel is stationary."""
        return self.k1.is_stationary() and self.k2.is_stationary()

    @property
    def requires_vector_input(self):
        """Return whether the kernel is stationary. """
        return (self.k1.requires_vector_input or
                self.k2.requires_vector_input)


class Sum(KernelOperator):
    """Sum of two kernels."""

    @property
    def pure_kernel_fn(self):
        """Kernel function of the two added kernels."""
        k1_fn = self.k1.pure_kernel_fn
        k2_fn = self.k2.pure_kernel_fn
        k1_dims = self.k1.n_dims

        @jit
        def kernel_fn(theta, x, y):
            return k1_fn(theta[:k1_dims], x, y) + k2_fn(theta[k1_dims:], x, y)

        return kernel_fn


class Product(KernelOperator):
    """Product of two kernels."""

    @property
    def pure_kernel_fn(self):
        """Kernel function of the two added kernels."""
        k1_fn = self.k1.pure_kernel_fn
        k2_fn = self.k2.pure_kernel_fn
        k1_dims = self.k1.n_dims

        @jit
        def kernel_fn(theta, x, y):
            return k1_fn(theta[:k1_dims], x, y) * k2_fn(theta[k1_dims:], x, y)

        return kernel_fn


class ConstantKernel(Kernel):
    """Kernel which always returns a constant."""

    def __init__(self, constant_value=1.0, constant_value_bounds=(1e-5, 1e5)):
        """Init kernel with constant_value."""
        self.constant_value = constant_value
        self.constant_value_bounds = constant_value_bounds

    @property
    def hyperparameter_constant_value(self):
        return Hyperparameter(
            "constant_value", "numeric", self.constant_value_bounds)

    @property
    def pure_kernel_fn(self):
        """Return the kernel fn."""
        @jit
        def kernel_fn(theta, x, y):
            return theta
        return kernel_fn


class Exponentiation(Kernel):
    """Exponentiation of a kernel."""

    def __init__(self, kernel: Kernel, exponent):
        """Init kernel exponentiation of kernel with exponent."""
        self.kernel = kernel
        self.exponent = exponent

    def is_stationary(self):
        """Whether kernel is stationary."""
        return self.kernel.is_stationary()

    def get_params(self, deep=True):
        """Get parameters of this kernel.

        Parameters:
            deep : boolean, optional
                If True, will return the parameters for this estimator and
                contained subobjects that are estimators.

        Returns:
            params : mapping of string to any
                Parameter names mapped to their values.

        """
        params = dict(kernel=self.kernel, exponent=self.exponent)
        if deep:
            deep_items = self.kernel.get_params().items()
            params.update(('kernel__' + k, val) for k, val in deep_items)
        return params

    @property
    def hyperparameters(self):
        """Return a list of all hyperparameter."""
        r = []
        for hyperparameter in self.kernel.hyperparameters:
            r.append(Hyperparameter("kernel__" + hyperparameter.name,
                                    hyperparameter.value_type,
                                    hyperparameter.bounds,
                                    hyperparameter.n_elements))
        return r

    @property
    def theta(self):
        """Return the (flattened, log-transformed) non-fixed hyperparameters.

        Note that theta are typically the log-transformed values of the
        kernel's hyperparameters as this representation of the search space
        is more amenable for hyperparameter search, as hyperparameters like
        length-scales naturally live on a log-scale.

        Returns:
            theta : array, shape (n_dims,)
                The non-fixed, log-transformed hyperparameters of the kernel

        """
        return self.kernel.theta

    @theta.setter
    def theta(self, theta):
        """Set the (flattened, log-transformed) non-fixed hyperparameters.

        Parameters:
            theta : array, shape (n_dims,)
                The non-fixed, log-transformed hyperparameters of the kernel

        """
        self.kernel.theta = theta

    @property
    def bounds(self):
        """Return the log-transformed bounds on the theta.

        Returns:
            bounds : array, shape (n_dims, 2)
                The log-transformed bounds on the kernel's hyperparameters
                theta

        """
        return self.kernel.bounds

    def __eq__(self, b):
        """Whether two instances are considered equal."""
        if type(self) != type(b):
            return False
        return (self.kernel == b.kernel and self.exponent == b.exponent)

    @property
    def pure_kernel_fn(self):
        """Pure kernel fn of exponentiated kernel."""
        exponent = self.exponent
        kernel_fn = self.kernel.pure_kernel_fn

        @jit
        def exp_kernel_fn(theta, x, y):
            return np.power(kernel_fn(theta, x, y), exponent)
        return exp_kernel_fn


class RBF(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """RBF Kernel."""

    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5)):
        """Initialize RBF kernel with length_scale and bounds."""
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds

    @property
    def anisotropic(self):
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    @property
    def hyperparameter_length_scale(self):
        if self.anisotropic:
            return Hyperparameter("length_scale", "numeric",
                                  self.length_scale_bounds,
                                  len(self.length_scale))
        return Hyperparameter(
            "length_scale", "numeric", self.length_scale_bounds)

    @property
    def pure_kernel_fn(self):
        """Pure kernel fn of RBF kernel."""
        @jit
        def kernel_fn(theta, x, y):
            # As we get a log-transformed theta as input, we need to transform
            # it back.
            diff = (x - y) / np.exp(theta)
            d = np.sum(diff ** 2, axis=-1)
            return np.exp(-0.5 * d)
        return kernel_fn

