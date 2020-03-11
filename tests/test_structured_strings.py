"""Test string kernels and utilities associated with them."""
import numpy as np
from sklearn_jax_kernels.structured.string_utils import AsciiBytesTransformer
from sklearn_jax_kernels.structured.strings import SpectrumKernel


class TestUtils:
    def test_ascii_bytes_transformer(self):
        strings = ['abc', 'def']
        transformer = AsciiBytesTransformer()
        trans = transformer.transform(strings)
        inverse = transformer.inverse_transform(trans)
        assert all([s1 == s2 for s1, s2 in zip(strings, inverse)])


class TestKernels:
    def test_spectrum_kernel(self):
        strings = ['aabbcc', 'aaabac']
        strings_transformed = AsciiBytesTransformer().transform(strings)
        kernel = SpectrumKernel(n_gram_length=2)
        K = kernel(strings_transformed)
        assert np.allclose(K, np.array([[5., 3.], [3., 7.]]))
