"""Implementation of string kernels."""
import jax.numpy as np
from jax.lax import dynamic_slice_in_dim
from jax.experimental import loops

from sklearn_jax_kernels import Kernel
from sklearn.gaussian_process.kernels import GenericKernelMixin


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

        if n_gram_length is None:
            # Assume input is kmer transformed
            def kernel_fn(theta, kmers1, kmers2):
                with loops.Scope() as s:
                    s.out = 0.
                    for i in s.range(kmers1.shape[0]):
                        kmer = kmers1[i]
                        is_same = np.all(kmer[None, :] == kmers2, axis=1)
                        n_matches = np.sum(is_same)
                        s.out += n_matches
                return s.out
        else:
            def kernel_fn(theta, string1, string2):
                with loops.Scope() as s1:
                    s1.out = 0.
                    for i in s1.range(len(string1) - n_gram_length + 1):
                        with loops.Scope() as s2:
                            s2.out = 0.
                            for j in s2.range(len(string2) - n_gram_length + 1):
                                substring1 = dynamic_slice_in_dim(
                                    string1, i, n_gram_length)
                                substring2 = dynamic_slice_in_dim(
                                    string2, j, n_gram_length)
                                s2.out += np.all(substring1 == substring2)
                        s1.out += s2.out
                return s1.out

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
