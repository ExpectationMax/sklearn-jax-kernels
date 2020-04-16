"""Base classes of Kernel implementation compatible with JAX."""
import abc
import numpy
from functools import partial
from sklearn.gaussian_process.kernels import Kernel as sklearn_kernel
from sklearn.gaussian_process.kernels import (
    Hyperparameter,
    StationaryKernelMixin,
    NormalizedKernelMixin
)
from jax import jit, vmap, value_and_grad
import jax.numpy as np
import jax.ops as ops
from jax.experimental import loops

from .config import config_value


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

    @staticmethod
    def _kernel_matrix_without_gradients(kernel_fn, theta, X, Y):
        kernel_fn = partial(kernel_fn, theta)
        if Y is None or (Y is X):
            if config_value('KERNEL_MATRIX_USE_LOOP'):
                n = len(X)
                with loops.Scope() as s:
                    # s.scattered_values = np.empty((n, n))
                    s.index1, s.index2 = np.tril_indices(n, k=0)
                    s.output = np.empty(len(s.index1))
                    for i in s.range(s.index1.shape[0]):
                        i1, i2 = s.index1[i], s.index2[i]
                        s.output = ops.index_update(
                            s.output,
                            i,
                            kernel_fn(X[i1], X[i2])
                        )
                first_update = ops.index_update(
                    np.empty((n, n)), (s.index1, s.index2), s.output)
                second_update = ops.index_update(
                    first_update, (s.index2, s.index1), s.output)
                return second_update
            else:
                n = len(X)
                values_scattered = np.empty((n, n))
                index1, index2 = np.tril_indices(n, k=-1)
                inst1, inst2 = X[index1], X[index2]
                values = vmap(kernel_fn)(inst1, inst2)
                values_scattered = ops.index_update(
                    values_scattered, (index1, index2), values)
                values_scattered = ops.index_update(
                    values_scattered, (index2, index1), values)
                values_scattered = ops.index_update(
                    values_scattered,
                    np.diag_indices(n),
                    vmap(lambda x: kernel_fn(x, x))(X)
                )
                return values_scattered
        else:
            if config_value('KERNEL_MATRIX_USE_LOOP'):
                with loops.Scope() as s:
                    s.output = np.empty((X.shape[0], Y.shape[0]))
                    for i in s.range(X.shape[0]):
                        x = X[i]
                        s.output = ops.index_update(
                            s.output,
                            i,
                            vmap(lambda y: kernel_fn(x, y))(Y)
                        )
                return s.output
            else:
                return vmap(lambda x: vmap(lambda y: kernel_fn(x, y))(Y))(X)

    @staticmethod
    def _kernel_matrix_with_gradients(kernel_fn, theta, X, Y):
        kernel_fn = value_and_grad(kernel_fn)
        kernel_fn = partial(kernel_fn, theta)
        if Y is None or (Y is X):
            if config_value('KERNEL_MATRIX_USE_LOOP'):
                n = len(X)
                with loops.Scope() as s:
                    s.scattered_values = np.empty((n, n))
                    s.scattered_grads = np.empty((n, n, len(theta)))
                    index1, index2 = np.tril_indices(n, k=0)
                    for i in s.range(index1.shape[0]):
                        i1, i2 = index1[i], index2[i]
                        value, grads = kernel_fn(X[i1], X[i2])
                        indexes = (np.stack([i1, i2]), np.stack([i2, i1]))
                        s.scattered_values = ops.index_update(
                            s.scattered_values,
                            indexes,
                            value
                        )
                        s.scattered_grads = ops.index_update(
                            s.scattered_grads, indexes, grads)
                return s.scattered_values, s.scattered_grads
            else:
                n = len(X)
                values_scattered = np.empty((n, n))
                grads_scattered = np.empty((n, n, len(theta)))
                index1, index2 = np.tril_indices(n, k=-1)
                inst1, inst2 = X[index1], X[index2]
                values, grads = vmap(kernel_fn)(inst1, inst2)
                # Scatter computed values into matrix
                values_scattered = ops.index_update(
                    values_scattered, (index1, index2), values)
                values_scattered = ops.index_update(
                    values_scattered, (index2, index1), values)
                grads_scattered = ops.index_update(
                    grads_scattered, (index1, index2), grads)
                grads_scattered = ops.index_update(
                    grads_scattered, (index2, index1), grads)
                diag_values, diag_grads = vmap(
                    lambda x: kernel_fn(x, x))(X)
                diag_indices = np.diag_indices(n)
                values_scattered = ops.index_update(
                    values_scattered, diag_indices, diag_values)
                grads_scattered = ops.index_update(
                    grads_scattered, diag_indices, diag_grads)
                return values_scattered, grads_scattered
        else:
            return vmap(
                lambda x: vmap(lambda y: kernel_fn(x, y))(Y))(X)

    def get_kernel_matrix_fn(self, eval_gradient):
        """Return pure function for computing kernel matrix and gradients.

        We do some internal caching in order to avoid recompiling the resulting
        function.

        Returned function has the signature: `f(theta, X, Y)`
        """
        cache_name = (
            '_cached_kernel_matrix_fn' + '_grad' if eval_gradient else '')
        if not hasattr(self, cache_name):
            pure_kernel_fn = self.pure_kernel_fn

            if eval_gradient:
                kernel_matrix_fn = jit(partial(
                    self._kernel_matrix_with_gradients,
                    pure_kernel_fn
                ))
            else:
                kernel_matrix_fn = jit(partial(
                    self._kernel_matrix_without_gradients,
                    pure_kernel_fn
                ))
            setattr(self, cache_name, kernel_matrix_fn)

        return getattr(self, cache_name)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Build kernel matrix from input data X and Y.

        Evtl. also compute the gradient with respect to the parameters.
        """
        X = np.asarray(X)
        if Y is not None:
            Y = np.asarray(Y)
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


class NormalizedKernel(NormalizedKernelMixin, Kernel):
    """Kernel wrapper which computes a normalized version of the kernel."""

    def __init__(self, kernel):
        self.kernel = kernel

    @property
    def pure_kernel_fn(self):
        """Not really needed in this particular case."""
        kernel_fn = self.kernel.pure_kernel_fn

        def wrapped(theta, x, y):
            K_xy = kernel_fn(theta, x, y)
            K_xx = kernel_fn(theta, x, x)
            K_yy = kernel_fn(theta, y, y)
            return K_xy / np.sqrt(K_xx * K_yy)
        return wrapped

    def get_kernel_matrix_fn(self, eval_gradient):
        """Return pure function for computing kernel matrix and gradients.

        We do some internal caching in order to avoid recompiling the resulting
        function. Further, we compute the output of the normalized kernel
        matrix at this stage in order to avoid recomputing the self similarity
        on each kernel evaluation.

        Returned function has the signature: `f(theta, X, Y)`
        """
        if config_value('NORMALIZED_KERNEL_PUSH_DOWN'):
            # In this case compute the normalization for each instance
            # inside the kernel fn. This recomputes the self similarities many
            # times, but does not require keeping multiple tensors of the size
            # of the kernel matrix in memory for computing normalization. This
            # is particularly the case when computing gradients with respect to
            # kernel parameters.
            return super().get_kernel_matrix_fn(eval_gradient)

        cache_name = '_kernel_matrix_fn' + '_grad' if eval_gradient else ''
        if not hasattr(self, cache_name):
            pure_kernel_fn = self.kernel.pure_kernel_fn
            if eval_gradient:
                kernel_matrix_with_grad = \
                    self._kernel_matrix_with_gradients

                def wrapped(theta, X, Y):
                    """Compute normalized kernel matrix and do chain rule."""
                    kmatrix, grads = kernel_matrix_with_grad(
                        pure_kernel_fn, theta, X, Y)
                    if Y is None:
                        diag = np.diag(kmatrix)
                        grad_diag_indices = np.diag_indices(kmatrix.shape[0])
                        diag_grad = grads[grad_diag_indices]
                        normalizer = np.sqrt(diag[:, None] * diag[None, :])
                        # Add dimensions for broadcasting
                        K_xx = diag[:, None, None]
                        K_yy = diag[None, :, None]
                        K_xx_grad = diag_grad[:, None, :]
                        K_yy_grad = diag_grad[None, :, :]

                        # Do the chain rule
                        grads = (
                            (
                                2 * K_xx * K_yy * grads -
                                kmatrix * (K_xx_grad * K_yy + K_xx * K_yy_grad)
                            ) / (2 * (K_xx * K_yy) ** (3/2))
                        )
                        return kmatrix / normalizer, grads
                    else:
                        # If y is not defined we need to compute the self
                        # similarity of each instance
                        kernel_fn_with_grad = partial(
                            value_and_grad(pure_kernel_fn), theta)
                        K_xx, K_xx_grad = vmap(
                            lambda x: kernel_fn_with_grad(x, x))(X)
                        K_yy, K_yy_grad = vmap(
                            lambda y: kernel_fn_with_grad(y, y))(Y)
                        # Add dimensions for broadcasting
                        K_xx = K_xx[:, None, None]
                        K_yy = K_yy[None, :, None]
                        K_xx_grad = K_xx_grad[:, None, :]
                        K_yy_grad = K_yy_grad[None, :, :]

                        normalizer = np.sqrt(K_xx[:, None] * K_yy[None, :])
                        # d/dw(k(x, y, w)/sqrt(k(x, x, w) k(y, y, w))) = (2
                        # k(x, x, w) k(y, y, w) k^(0, 0, 1)(x, y, w) - k(x, y,
                        # w) (k^(0, 0, 1)(x, x, w) k(y, y, w) + k(x, x, w)
                        # k^(0, 0, 1)(y, y, w)))/(2 (k(x, x, w) k(y, y,
                        # w))^(3/2))
                        grads = (
                            (
                                2 * K_xx * K_yy * grads -
                                kmatrix * (K_xx_grad * K_yy + K_xx * K_yy_grad)
                            ) /
                            (2 * (K_xx * K_yy) ** (3/2))
                        )

                        return kmatrix / normalizer, grads

                kernel_matrix_fn = jit(wrapped)
            else:
                kernel_matrix = self._kernel_matrix_without_gradients

                def wrapped(theta, X, Y):
                    """Compute normalized kernel matrix."""
                    kmatrix = kernel_matrix(pure_kernel_fn, theta, X, Y)
                    if Y is None:
                        diag = np.diag(kmatrix)
                        normalizer = np.sqrt(diag[:, None] * diag[None, :])
                        return kmatrix / normalizer
                    else:
                        # If y is not defined we need to compute the self
                        # similarity of each instance
                        K_xx = vmap(lambda x: pure_kernel_fn(theta, x, x))(X)
                        K_yy = vmap(lambda y: pure_kernel_fn(theta, y, y))(Y)
                        normalizer = np.sqrt(K_xx[:, None] * K_yy[None, :])
                        return kmatrix / normalizer

                kernel_matrix_fn = jit(wrapped)
            setattr(self, cache_name, kernel_matrix_fn)
        return getattr(self, cache_name)

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
        params = dict(kernel=self.kernel)
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
        return self.kernel == b.kernel


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

        def kernel_fn(theta, x, y):
            return k1_fn(theta[:k1_dims], x, y) * k2_fn(theta[k1_dims:], x, y)

        return kernel_fn

    def __repr__(self):
        """Return representation of kernel."""
        return "{0} * {1}".format(self.k1, self.k2)


class ConstantKernel(StationaryKernelMixin, Kernel):
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
        if self.hyperparameter_constant_value.fixed:
            value = self.constant_value

            def kernel_fn(theta, x, y):
                return value
        else:
            def kernel_fn(theta, x, y):
                return np.exp(theta[0])  # Theta is in log domain and array

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
        if self.hyperparameter_length_scale.fixed:
            length_scale = self.length_scale
            if np.iterable(length_scale):
                # handle case when length scale is fixed and provided as list
                length_scale = np.asarray(length_scale)

            def kernel_fn(theta, x, y):
                # as we get a log-transformed theta as input, we need to transform
                # it back.
                diff = (x - y) / length_scale
                d = np.sum(diff ** 2, axis=-1)
                return np.exp(-0.5 * d)
        else:
            def kernel_fn(theta, x, y):
                # as we get a log-transformed theta as input, we need to transform
                # it back.
                diff = (x - y) / np.exp(theta)
                d = np.sum(diff ** 2, axis=-1)
                return np.exp(-0.5 * d)
        return kernel_fn

    def __repr__(self):
        if self.anisotropic:
            return "{0}(length_scale=[{1}])".format(
                self.__class__.__name__, ", ".join(map("{0:.3g}".format,
                                                   self.length_scale)))
        else:  # isotropic
            return "{0}(length_scale={1:.3g})".format(
                self.__class__.__name__, np.ravel(self.length_scale)[0])
