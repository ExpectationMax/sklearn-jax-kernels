"""Test string kernels and utilities associated with them."""
import numpy as np
from sklearn_jax_kernels.structured.string_utils import (
    AsciiBytesTransformer, CompressAlphabetTransformer, NGramTransformer)
from sklearn_jax_kernels import RBF
from sklearn_jax_kernels.structured.strings import (
    DistanceSpectrumKernel,
    DistanceFromEndSpectrumKernel,
    RevComplementSpectrumKernel,
    SpectrumKernel
)
# from jax import config
# config.update('jax_disable_jit', True)


class TestUtils:
    def test_ascii_bytes_transformer(self):
        strings = ['abc', 'def']
        transformer = AsciiBytesTransformer()
        trans = transformer.transform(strings)
        inverse = transformer.inverse_transform(trans)
        assert all([s1 == s2 for s1, s2 in zip(strings, inverse)])

    def test_ngram_transformer(self):
        strings = np.asarray([list('abcde')])
        ngrams = np.asarray([[
            list('abc'),
            list('bcd'),
            list('cde')
        ]])
        transformer = NGramTransformer(3)
        transformed = transformer.transform(strings)
        assert np.all(np.ravel(ngrams) == np.ravel(transformed))

    def test_compress_alphabet_transformer(self):
        strings = np.asarray(['abc'])
        transf = CompressAlphabetTransformer()
        transf.fit(strings)
        out = transf.transform(np.asarray(['cbad']))
        assert np.all(np.array([[2, 1, 0, 3]]) == out)


class TestKernels:
    def test_spectrum_kernel_example(self):
        strings = ['aabbcc', 'aaabac']
        strings_transformed = AsciiBytesTransformer().transform(strings)
        kernel = SpectrumKernel(n_gram_length=2)
        K = kernel(strings_transformed)
        assert np.allclose(K, np.array([[5., 3.], [3., 7.]]))

    def test_spectrum_kernel_ngram_transform(self):
        n_gram_length = 2
        strings = ['aabbcc', 'aaabac']
        strings_transformed = AsciiBytesTransformer().transform(strings)
        ngrams = NGramTransformer(n_gram_length).transform(strings_transformed)

        kernel_strings = SpectrumKernel(n_gram_length=n_gram_length)
        kernel_ngrams = SpectrumKernel(n_gram_length=None)
        K_strings = kernel_strings(strings_transformed)
        K_ngrams = kernel_ngrams(ngrams)
        assert np.allclose(K_strings, K_ngrams)

    def test_distance_spectrum_kernel_ngram_transform(self):
        n_gram_length = 2
        distance_kernel = RBF(1.0)
        strings = ['aabbcc', 'aaabac']
        strings_transformed = AsciiBytesTransformer().transform(strings)
        ngrams = NGramTransformer(n_gram_length).transform(strings_transformed)

        kernel_strings = DistanceSpectrumKernel(distance_kernel, n_gram_length)
        kernel_ngrams = DistanceSpectrumKernel(distance_kernel, None)
        K_strings = kernel_strings(strings_transformed)
        K_ngrams = kernel_ngrams(ngrams)
        assert np.allclose(K_strings, K_ngrams)

    def test_distance_spectrum_kernel(self):
        distance_kernel = RBF(1.0)
        strings = ['aabbcc', 'aaabac']
        strings_transformed = AsciiBytesTransformer().transform(strings)
        kernel = DistanceSpectrumKernel(distance_kernel, 2)
        K = kernel(strings_transformed)
        K_gt = np.array([
             [5.,        2.2130613],
             [2.2130613, 6.2130613]
        ])
        assert np.allclose(K, K_gt)

    def test_distance_from_end_spectrum_kernel(self):
        distance_kernel = RBF(1.0)
        strings = ['abc', 'cba']
        strings_transformed = AsciiBytesTransformer().transform(strings)
        kernel = DistanceFromEndSpectrumKernel(distance_kernel, 1)
        K = kernel(strings_transformed)
        K_gt = np.array([
             [3., 3.],
             [3., 3.]
        ])
        assert np.allclose(K, K_gt)

    def test_rev_comp_spectrum_kernel(self):
        mapping = np.array([1, 0, 3, 2], np.uint8)
        strings = np.array([[0, 1, 2, 3], [2, 3, 0, 1]], np.uint8)
        kernel = RevComplementSpectrumKernel(2, mapping)
        K = kernel(strings)
        print(K)
        K_gt = np.array([
             [5., 5.],
             [5., 5.]
        ])
        assert np.allclose(K, K_gt)
