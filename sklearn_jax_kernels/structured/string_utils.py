"""Utilities for usage with strings."""
import numpy as np
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
            return np.asarray([bytearray(x, 'ascii', 'strict') for x in X])

        # Varying length
        return [np.asarray(bytearray(a, 'ascii', 'strict')) for a in X]

    def inverse_transform(self, X):
        return [str(bytes(a), 'ascii', 'strict') for a in X]


class NGramTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, ngram_length):
        self.ngram_length = ngram_length

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        n = X.shape[1]
        ngram_slices = [
            X[:, i:n+1-self.ngram_length+i]
            for i in range(0, self.ngram_length)
        ]
        return np.stack(ngram_slices, axis=1)
