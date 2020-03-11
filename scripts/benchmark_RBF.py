"""Compare performance of jax kernel with default sklearn kernel."""
import timeit

import numpy as np
import jax.numpy as jnp
from sklearn import datasets
from sklearn.gaussian_process import GaussianProcessClassifier as sk_GPC
from sklearn.gaussian_process.kernels import RBF as sklearn_RBF
from sklearn_jax_kernels import RBF as jax_RBF
from sklearn_jax_kernels import GaussianProcessClassifier as jax_GPC

# import some data to play with
digits = datasets.load_digits()
X = digits.data
y = np.array(digits.target, dtype=int)

X_jax = jnp.asarray(X)
y_jax = jnp.asarray(y)

sk_kernel = 1.0 * sklearn_RBF([1.0])
jax_kernel = 1.0 * jax_RBF([1.0])

sk_clf = sk_GPC(kernel=sk_kernel, copy_X_train=False)
jax_clf = jax_GPC(kernel=jax_kernel, copy_X_train=False)

sk_clf.fit(X, y)
jax_clf.fit(X_jax, y_jax)

def fit_with_sklearn_kernel():
    sk_clf.fit(X, y)

def fit_with_jax_kernel():
    jax_clf.fit(X_jax, y_jax)

time_sk = timeit.timeit(fit_with_sklearn_kernel, number=1)
print(time_sk)
time_jax = timeit.timeit(fit_with_jax_kernel, number=1)
print(time_jax)
