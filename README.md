# sklearn-jax-kernels

[![Build Status](https://travis-ci.com/ExpectationMax/sklearn-jax-kernels.svg?token=3sUUnmMzs9wxN3Qapssj&branch=master)](https://travis-ci.com/ExpectationMax/sklearn-jax-kernels)

**Warning: This project is still in an early stage it could be that the API
will change in the future, further functionality is still very limited to the
use cases which defined the creation of the project (application to DNA
sequences present in Biology).**

## Why?
Ever wanted to run a kernel-based model from
[scikit-learn](https://scikit-learn.org/) on a relatively large dataset?  If so
you will have noticed, that this can take extraordinarily long and require huge
amounts of memory, especially if you are using compositions of kernels (such as
for example `k1 * k2 + k3`).  This is due to the way Kernels are computed in
scikit-learn: For each kernel, the complete kernel matrix is computed, and the
compositions are then computed from the kernel matrices.  Further,
`scikit-learn` does not rely on an automatic differentiation framework for the
computation of gradients though kernel operations.

## Introduction

`sklearn-jax-kernels` was designed to circumvent these issues:

 - The utilization of [JAX](https://github.com/google/jax) allows accelerating
   kernel computations through [XLA](https://www.tensorflow.org/xla)
   optimizations, computation on GPUs and simplifies the computation of
   gradients though kernels
 - The composition of kernels takes place on a per-element basis, such that
   unnecessary copies can be optimized away by JAX compilation

The goal of `sklearn-jax-kernels` is to provide the same flexibility and ease
of use as known from `scikit-learn` kernels while improving speed and allowing
the faster design of new kernels through Automatic Differentiation.

The kernels in this package follow the [scikit-learn kernel
API](https://scikit-learn.org/stable/modules/gaussian_process.html#gaussian-process-kernel-api).

## Quickstart

A short demonstration of how the kernels can be used, inspired by the
[ scikit-learn
documentation](https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpc_iris.html).

```python
from sklearn import datasets
import jax.numpy as jnp
from sklearn_jax_kernels import RBF, GaussianProcessClassifier

iris = datasets.load_iris()
X = jnp.asarray(iris.data)
y = jnp.array(iris.target, dtype=int)

kernel = 1. + RBF(length_scale=1.0)
gpc = GaussianProcessClassifier(kernel=kernel).fit(X, y)
```

Here a further example demonstrating how kernels can be combined:

```python
from sklearn_jax_kernels.base_kernels import RBF, NormalizedKernel
from sklearn_jax_kernels.structured.strings import SpectrumKernel

my_kernel = RBF(1.) * SpectrumKernel(n_gram_length=3)
my_kernel_2 = RBF(1.) + RBF(2.)
my_kernel_2 = NormalizedKernel(my_kernel_2)
```

Some further inspiration can be taken from the tests in the subfolder `tests`.

## Implemented Kernels

 - Kernel compositions ($+,-,*,/$, exponentiation)
 - Kernels for real valued data:  
     - RBF kernel
 - Kernels for same length strings:  
     - SpectrumKernel
     - DistanceSpectrumKernel, SpectrumKernel with distance weight between
       matching substrings
     - ReverseComplement Spectrum kernel (relevant for applications in Biology
       when working with DNA sequences)

## TODOs

 - Implement more fundamental Kernels
 - Implement jax compatible version of GaussianProcessRegressor
 - Optimize GaussianProcessClassifier for performance
 - Run benchmarks to show benefits in speed
 - Add fake "split" kernel which allows to apply different kernels to different
   parts of the input
