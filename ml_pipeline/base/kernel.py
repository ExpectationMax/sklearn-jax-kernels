"""Kernel implementation compatible with JAX."""
import abc
from functools import partial
from sklearn.gaussian_process.kernels import Kernel as sklearn_kernel
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
        fn(parameters_dict, x1, x2).
        """

    def get_kernel_matrix_fn(self, eval_gradient):
        """Return pure function for computing kernel matrix and gradients.

        Returned function has the signature: `f(params, X, Y)`
        """
        pure_kernel_fn = self.pure_kernel_fn

        def kernel_matrix_fn(params, X, Y):
            kernel_fn = pure_kernel_fn
            if eval_gradient:
                kernel_fn = value_and_grad(pure_kernel_fn)
            kernel_fn = jit(partial(kernel_fn, params))
            return vmap(lambda x: vmap(lambda y: kernel_fn(x, y))(Y))(X)

        return kernel_matrix_fn

    def __call__(self, X, Y=None, eval_gradient=False):
        """Build kernel matrix from input data X and Y.

        Evtl. also compute the gradient with respect to the parameters.
        """
        if Y is None:
            Y = X

        params = self.get_params()

        return self.get_kernel_matrix_fn(eval_gradient)(params, X, Y)

    def diag(self, X):
        params = self.get_params()
        return vmap(lambda x: self.pure_kernel_fn(params, x, x))(X)

    def is_stationary(self):
        return False

    def __add__(self, b):
        if not isinstance(b, Kernel):
            return Sum(self, ConstantKernel(b))
        return Sum(self, b)

    def __radd__(self, b):
        if not isinstance(b, Kernel):
            return Sum(ConstantKernel(b), self)
        return Sum(b, self)

    def __mul__(self, b):
        if not isinstance(b, Kernel):
            return Product(self, ConstantKernel(b))
        return Product(self, b)

    def __rmul__(self, b):
        if not isinstance(b, Kernel):
            return Product(ConstantKernel(b), self)
        return Product(b, self)

    def __pow__(self, b):
        return Exponentiation(self, b)


class KernelOperator(Kernel):
    """Base class for operations on kernels."""

    def __init__(self, k1: Kernel, k2: Kernel):
        """Compute an operation between two kernels k1, and k2."""
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
        params = dict()
        if deep:
            deep_items = self.k1.get_params().items()
            params.update(('k1__' + k, val) for k, val in deep_items)
            deep_items = self.k2.get_params().items()
            params.update(('k2__' + k, val) for k, val in deep_items)

        return params

    @property
    def pure_kernel_fn(self):
        """Not implmented."""
        raise NotImplementedError

    @staticmethod
    def split_kernel_parameters(params):
        """Split kernel parameters for k1 and k2.

        Parameters:
            params: Dict with parameters including kernel prefixes

        Returns:
            (k1_params, k2_params)

        """
        k1_params = {}
        k2_params = {}
        for name, value in params.items():
            name_without_prefix = name[4:]
            if name.startswith('k1__'):
                k1_params[name_without_prefix] = value
            else:
                k2_params[name_without_prefix] = value
        return k1_params, k2_params


class Sum(KernelOperator):
    """Sum of two kernels."""

    @property
    def pure_kernel_fn(self):
        """Kernel function of the two added kernels."""
        k1_fn = self.k1.pure_kernel_fn
        k2_fn = self.k2.pure_kernel_fn

        def kernel_fn(params, x, y):
            k1_params, k2_params = self.split_kernel_parameters(params)
            return k1_fn(k1_params, x, y) + k2_fn(k2_params, x, y)

        return kernel_fn


class Product(KernelOperator):
    """Product of two kernels."""

    @property
    def pure_kernel_fn(self):
        """Kernel function of the two added kernels."""
        k1_fn = self.k1.pure_kernel_fn
        k2_fn = self.k2.pure_kernel_fn

        def kernel_fn(params, x, y):
            k1_params, k2_params = self.split_kernel_parameters(params)
            return k1_fn(k1_params, x, y) * k2_fn(k2_params, x, y)

        return kernel_fn


class ConstantKernel(Kernel):
    """Kernel which always returns a constant."""

    def __init__(self, constant_value):
        self.constant_value = constant_value

    @property
    def pure_kernel_fn(self):
        """Return the kernel fn."""
        def kernel_fn(params, x, y):
            return params['constant_value']
        return kernel_fn


class Exponentiation(Kernel):
    def __init__(self, kernel: Kernel, exponent):
        self.kernel = kernel
        self.exponent = exponent

    def get_params(self, deep=True):
        """Get parameters of this kernel.
        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        params = dict(kernel=self.kernel, exponent=self.exponent)
        if deep:
            deep_items = self.kernel.get_params().items()
            params.update(('kernel__' + k, val) for k, val in deep_items)
        return params

    @staticmethod
    def _get_kernel_params(params):
        return {
            name[8:]: param for name, param in params.items()
            if name.startswith('kernel__')
        }

    @property
    def pure_kernel_fn(self):
        """Pure kernel fn of exponentiated kernel."""
        get_kernel_params = self._get_kernel_params
        kernel = self.kernel

        def kernel_fn(params, x, y):
            exponent = params['exponent']
            kernel_params = get_kernel_params(params)
            return np.pow(kernel.pure_kernel_fn(kernel_params, x, y), exponent)
        return kernel_fn


class RBF(Kernel):
    """RBF Kernel."""

    def __init__(self, length_scale):
        self.length_scale = length_scale

    @property
    def pure_kernel_fn(self):
        def kernel_fn(params, x, y):
            d = np.sum((x - y) ** 2, axis=-1)
            return np.exp(-0.5 * d / params['length_scale'])
        return kernel_fn
