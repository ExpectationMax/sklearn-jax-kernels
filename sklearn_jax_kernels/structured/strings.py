"""Implementation of string kernels."""
from functools import partial
from jax import device_put, vmap
import jax.numpy as np
from jax.lax import dynamic_slice_in_dim
from jax.experimental import loops

from sklearn_jax_kernels import Kernel
from sklearn.gaussian_process.kernels import GenericKernelMixin, Hyperparameter


class SpectrumKernel(GenericKernelMixin, Kernel):
    """Spectrum string kernel.

    As described in:
    ```
    @incollection{leslie2001spectrum,
      title={
        The spectrum kernel: A string kernel for SVM protein classification},
      author={
        Leslie, Christina and Eskin, Eleazar and Noble, William Stafford},
      booktitle={Biocomputing 2002},
      pages={564--575},
      year={2001},
      publisher={World Scientific}
    }
    ```
    """

    def __init__(self, n_gram_length):
        """Spectrum kernel on strings.

        Assumes input was transformed via `AcsiiBytesTransformer` or similar
        tranformation into a jax compatible datatype.

        Parameters:
            n_gram_length: Length of ngrams to compare. If `None` it is assumed
                that the input is 2d where the final axis is the n_grams.

        """
        self.n_gram_length = n_gram_length

    @property
    def pure_kernel_fn(self):
        """Return the pure fuction for computing the kernel."""
        n_gram_length = self.n_gram_length

        def kmer_kernel_fn(theta, kmers1, kmers2):
            same_kmer = np.all(kmers1[None, :, :] == kmers2[:, None, :], axis=2)
            return np.sum(same_kmer)

        if n_gram_length is None:
            # Assume input is kmer transformed
            kernel_fn = kmer_kernel_fn
        else:
            def kernel_fn(theta, string1, string2):
                def make_to_kmers(string):
                    ngram_slices = [
                        string[i:len(string)+1-n_gram_length+i]
                        for i in range(0, n_gram_length)
                    ]
                    return np.stack(ngram_slices, axis=1)
                kmers1 = make_to_kmers(string1)
                kmers2 = make_to_kmers(string2)
                return kmer_kernel_fn(theta, kmers1, kmers2)
        return kernel_fn

    @property
    def hyperparameters(self):
        """Return a list of all hyperparameter."""
        return []

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
        return np.empty((0,))

    @theta.setter
    def theta(self, theta):
        """Set the (flattened, log-transformed) non-fixed hyperparameters.

        Parameters:
            theta : array, shape (n_dims,)
                The non-fixed, log-transformed hyperparameters of the kernel

        """

    @property
    def bounds(self):
        """Return the log-transformed bounds on the theta.

        Returns:
            bounds : array, shape (n_dims, 2)
                The log-transformed bounds on the kernel's hyperparameters
                theta

        """
        return np.empty((0, 2))

    def is_stationary(self):
        """Whether this kernel is stationary."""
        return False


class DistanceSpectrumKernel(Kernel):
    """Spectrum kernel weighting ngrams by their distance in the sequence."""

    def __init__(self, distance_kernel: Kernel, n_gram_length):
        """Initialize DistanceSpectrumKernel using distance_kernel.

        Args:
            distance_kernel (Kernel): Kernel used to quantify distance of kmers
                in the string.

        """
        self.distance_kernel = distance_kernel
        self.n_gram_length = n_gram_length

    @property
    def pure_kernel_fn(self):
        """Return the pure function for computing the kernel."""
        n_gram_length = self.n_gram_length

        distance_kernel = self.distance_kernel.pure_kernel_fn

        def kmer_kernel_fn(theta, kmers1, kmers2):
            pos_kernel = partial(distance_kernel, theta)
            same_kmer = np.all(
                kmers1[:, None, :] == kmers2[None, :, :],
                axis=2
            )
            offsets1 = np.arange(kmers1.shape[0])
            offsets2 = np.arange(kmers2.shape[0])
            distance_weight = vmap(
                lambda i: vmap(lambda j: pos_kernel(i, j))(offsets2))(offsets1)
            return np.sum(same_kmer * distance_weight)

        if n_gram_length is None:
            # Assume input is kmer transformed
            kernel_fn = kmer_kernel_fn
        else:
            def kernel_fn(theta, string1, string2):
                def make_to_kmers(string):
                    ngram_slices = [
                        string[i:len(string)+1-n_gram_length+i]
                        for i in range(0, n_gram_length)
                    ]
                    return np.stack(ngram_slices, axis=1)
                kmers1 = make_to_kmers(string1)
                kmers2 = make_to_kmers(string2)
                return kmer_kernel_fn(theta, kmers1, kmers2)

        return kernel_fn

    @property
    def hyperparameters(self):
        """Return a list of all hyperparameter."""
        r = []
        for hyperparameter in self.distance_kernel.hyperparameters:
            r.append(Hyperparameter("distance_kernel__" + hyperparameter.name,
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
        return self.distance_kernel.theta

    @theta.setter
    def theta(self, theta):
        """Set the (flattened, log-transformed) non-fixed hyperparameters.

        Parameters:
            theta : array, shape (n_dims,)
                The non-fixed, log-transformed hyperparameters of the kernel

        """
        self.distance_kernel.theta = theta

    @property
    def bounds(self):
        """Return the log-transformed bounds on the theta.

        Returns:
            bounds : array, shape (n_dims, 2)
                The log-transformed bounds on the kernel's hyperparameters
                theta

        """
        return self.distance_kernel.bounds

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
        params = dict(
            distance_kernel=self.distance_kernel,
            n_gram_length=self.n_gram_length
        )
        if deep:
            deep_items = self.distance_kernel.get_params().items()
            params.update(
                ('distance_kernel__' + k, val) for k, val in deep_items)
        return params

    def __eq__(self, b):
        """Whether two instances are considered equal."""
        if type(self) != type(b):
            return False
        return (
            self.distance_kernel == b.distance_kernel and
            self.n_gram_length == b.n_gram_length
        )

    def is_stationary(self):
        """Whether this kernel is stationary."""
        return False


class DistanceFromEndSpectrumKernel(Kernel):
    """Spectrum kernel weighting ngrams by their distance in the sequence."""

    def __init__(self, distance_kernel: Kernel, n_gram_length):
        """Initialize DistanceSpectrumKernel using distance_kernel.

        Args:
            distance_kernel (Kernel): Kernel used to quantify distance of kmers
                in the string.

        """
        self.distance_kernel = distance_kernel
        self.n_gram_length = n_gram_length

    @property
    def pure_kernel_fn(self):
        """Return the pure fuction for computing the kernel."""
        n_gram_length = self.n_gram_length

        distance_kernel = self.distance_kernel.pure_kernel_fn

        def kmer_kernel_fn(theta, kmers1, kmers2):
            pos_kernel = partial(distance_kernel, theta)
            kmer2_offsets = np.arange(kmers1.shape[0], dtype=np.uint32)
            distances_from_end = np.min(
                np.stack([kmer2_offsets, kmer2_offsets[::-1]], axis=0), axis=0)

            with loops.Scope() as s:
                s.out = 0.
                for i in s.range(kmers1.shape[0]):
                    distance_from_end = distances_from_end[i]
                    kmer = kmers1[i]
                    distances = vmap(
                        lambda j: pos_kernel(distance_from_end, j))(distances_from_end)
                    is_same = np.all(kmer[None, :] == kmers2, axis=1)
                    n_matches = np.sum(is_same * distances)
                    s.out += n_matches
            return s.out
        if n_gram_length is None:
            # Assume input is kmer transformed
            kernel_fn = kmer_kernel_fn
        else:
            def kernel_fn(theta, string1, string2):
                def make_to_kmers(string):
                    ngram_slices = [
                        string[i:len(string)+1-n_gram_length+i]
                        for i in range(0, n_gram_length)
                    ]
                    return np.stack(ngram_slices, axis=1)
                kmers1 = make_to_kmers(string1)
                kmers2 = make_to_kmers(string2)
                return kmer_kernel_fn(theta, kmers1, kmers2)

        return kernel_fn

    @property
    def hyperparameters(self):
        """Return a list of all hyperparameter."""
        r = []
        for hyperparameter in self.distance_kernel.hyperparameters:
            r.append(Hyperparameter("distance_kernel__" + hyperparameter.name,
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
        return self.distance_kernel.theta

    @theta.setter
    def theta(self, theta):
        """Set the (flattened, log-transformed) non-fixed hyperparameters.

        Parameters:
            theta : array, shape (n_dims,)
                The non-fixed, log-transformed hyperparameters of the kernel

        """
        self.distance_kernel.theta = theta

    @property
    def bounds(self):
        """Return the log-transformed bounds on the theta.

        Returns:
            bounds : array, shape (n_dims, 2)
                The log-transformed bounds on the kernel's hyperparameters
                theta

        """
        return self.distance_kernel.bounds

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
        params = dict(
            distance_kernel=self.distance_kernel,
            n_gram_length=self.n_gram_length
        )
        if deep:
            deep_items = self.distance_kernel.get_params().items()
            params.update(
                ('distance_kernel__' + k, val) for k, val in deep_items)
        return params

    def __eq__(self, b):
        """Whether two instances are considered equal."""
        if type(self) != type(b):
            return False
        return (
            self.distance_kernel == b.distance_kernel and
            self.n_gram_length == b.n_gram_length
        )

    def is_stationary(self):
        """Whether this kernel is stationary."""
        return False


class RevComplementSpectrumKernel(GenericKernelMixin, Kernel):
    """Spectrum string kernel which also count reverse complement matches."""

    def __init__(self, n_gram_length, mapping):
        """Spectrum kernel on strings.

        Assumes input was transformed via `AcsiiBytesTransformer` or similar
        tranformation into a jax compatible datatype.

        Parameters:
            n_gram_length: Length of ngrams to compare. If `None` it is assumed
                that the input is 2d where the final axis is the n_grams.
            mapping: Array of length of alphabet which defines what is
                considered the complement to a particular character

        """
        self.n_gram_length = n_gram_length
        self.mapping = mapping

    @property
    def pure_kernel_fn(self):
        """Return the pure fuction for computing the kernel."""
        n_gram_length = self.n_gram_length
        mapping = device_put(np.array(self.mapping))

        def kmer_kernel_fn(theta, kmers1, kmers2):
            with loops.Scope() as s:
                s.out = 0.
                for i in s.range(kmers1.shape[0]):
                    kmer = kmers1[i]
                    rev_comp = mapping[kmer][::-1]
                    is_same_fw = np.all(kmer[None, :] == kmers2, axis=1)
                    is_same_rev_comp = np.all(
                        rev_comp[None, :] == kmers2, axis=1)
                    n_matches = np.sum(is_same_fw) + np.sum(is_same_rev_comp)
                    s.out += n_matches
            return s.out

        if n_gram_length is None:
            # Assume input is kmer transformed
            kernel_fn = kmer_kernel_fn
        else:
            def kernel_fn(theta, string1, string2):
                def make_to_kmers(string):
                    ngram_slices = [
                        string[i:len(string)+1-n_gram_length+i]
                        for i in range(0, n_gram_length)
                    ]
                    return np.stack(ngram_slices, axis=1)
                kmers1 = make_to_kmers(string1)
                kmers2 = make_to_kmers(string2)
                return kmer_kernel_fn(theta, kmers1, kmers2)
        return kernel_fn

    @property
    def hyperparameters(self):
        """Return a list of all hyperparameter."""
        return []

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
        return np.empty((0,))

    @theta.setter
    def theta(self, theta):
        """Set the (flattened, log-transformed) non-fixed hyperparameters.

        Parameters:
            theta : array, shape (n_dims,)
                The non-fixed, log-transformed hyperparameters of the kernel

        """

    @property
    def bounds(self):
        """Return the log-transformed bounds on the theta.

        Returns:
            bounds : array, shape (n_dims, 2)
                The log-transformed bounds on the kernel's hyperparameters
                theta

        """
        return np.empty((0, 2))

    def is_stationary(self):
        """Whether this kernel is stationary."""
        return False


class DistanceRevComplementSpectrumKernel(Kernel):
    """Spectrum kernel weighting ngrams by their distance in the sequence."""

    def __init__(self, distance_kernel: Kernel, n_gram_length, mapping):
        """Initialize DistanceSpectrumKernel using distance_kernel.

        Args:
            distance_kernel (Kernel): Kernel used to quantify distance of kmers
                in the string.

        """
        self.distance_kernel = distance_kernel
        self.n_gram_length = n_gram_length
        self.mapping = mapping

    @property
    def pure_kernel_fn(self):
        """Return the pure function for computing the kernel."""
        n_gram_length = self.n_gram_length

        distance_kernel = self.distance_kernel.pure_kernel_fn

        def kmer_kernel_fn(theta, kmers1, kmers2):
            mapping = device_put(self.mapping)
            pos_kernel = partial(distance_kernel, theta)

            rev_complement = np.reshape(
                mapping[np.ravel(kmers2)], kmers2.shape)[:, ::-1]

            same_kmer = np.all(
                kmers1[:, None, :] == kmers2[None, :, :],
                axis=2
            )
            same_rev_comp = np.all(
                kmers1[:, None, :] == rev_complement[None, :, :],
                axis=2
            )
            offsets1 = np.arange(kmers1.shape[0])
            offsets2 = np.arange(kmers2.shape[0])
            weight = vmap(
                lambda i: vmap(lambda j: pos_kernel(i, j))(offsets2))(offsets1)
            weight_rev_comp = vmap(
                lambda i: vmap(lambda j: pos_kernel(i, j))(offsets2[::-1]))(offsets1)

            return (
                np.sum(same_kmer * weight) +
                np.sum(same_rev_comp * weight_rev_comp)
            )

        if n_gram_length is None:
            # Assume input is kmer transformed
            kernel_fn = kmer_kernel_fn
        else:
            def kernel_fn(theta, string1, string2):
                def make_to_kmers(string):
                    ngram_slices = [
                        string[i:len(string)+1-n_gram_length+i]
                        for i in range(0, n_gram_length)
                    ]
                    return np.stack(ngram_slices, axis=1)
                kmers1 = make_to_kmers(string1)
                kmers2 = make_to_kmers(string2)
                return kmer_kernel_fn(theta, kmers1, kmers2)

        return kernel_fn

    @property
    def hyperparameters(self):
        """Return a list of all hyperparameter."""
        r = []
        for hyperparameter in self.distance_kernel.hyperparameters:
            r.append(Hyperparameter("distance_kernel__" + hyperparameter.name,
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
        return self.distance_kernel.theta

    @theta.setter
    def theta(self, theta):
        """Set the (flattened, log-transformed) non-fixed hyperparameters.

        Parameters:
            theta : array, shape (n_dims,)
                The non-fixed, log-transformed hyperparameters of the kernel

        """
        self.distance_kernel.theta = theta

    @property
    def bounds(self):
        """Return the log-transformed bounds on the theta.

        Returns:
            bounds : array, shape (n_dims, 2)
                The log-transformed bounds on the kernel's hyperparameters
                theta

        """
        return self.distance_kernel.bounds

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
        params = dict(
            distance_kernel=self.distance_kernel,
            n_gram_length=self.n_gram_length,
            mapping=self.mapping
        )
        if deep:
            deep_items = self.distance_kernel.get_params().items()
            params.update(
                ('distance_kernel__' + k, val) for k, val in deep_items)
        return params

    def __eq__(self, b):
        """Whether two instances are considered equal."""
        if type(self) != type(b):
            return False
        return (
            self.distance_kernel == b.distance_kernel and
            self.n_gram_length == b.n_gram_length
        )

    def is_stationary(self):
        """Whether this kernel is stationary."""
        return False
