"""Utilities for usage with strings."""
from functools import partial
import numpy as np
import jax.numpy as jnp
from jax import vmap
from sklearn.base import BaseEstimator, TransformerMixin


class AsciiBytesTransformer(TransformerMixin):
    """Convert python strings into ascii byte arrays.

    This allows them to be used with jax.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if len(np.unique([len(x) for x in X])) == 1:
            # All strings are of same length
            return jnp.asarray([bytearray(x, 'ascii', 'strict') for x in X])

        # Varying length
        return [jnp.asarray(bytearray(a, 'ascii', 'strict')) for a in X]

    def inverse_transform(self, X):
        return [str(bytes(a), 'ascii', 'strict') for a in X]


class NGramTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, ngram_length):
        self.ngram_length = ngram_length

    @staticmethod
    def build_kmers(nmers, string):
        n = string.shape[0]
        return jnp.stack(
            [string[i:1+n+i-nmers] for i in range(0, nmers)], axis=1)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return vmap(partial(self.build_kmers, self.ngram_length))(X)
