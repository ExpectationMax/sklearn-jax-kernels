"""Subclass of sklearn Gaussian process classifier using JAX."""
from functools import partial
from sklearn.gaussian_process import GaussianProcessClassifier as GPC
from sklearn.gaussian_process._gpc import (
    _BinaryGaussianProcessClassifierLaplace)
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.utils.validation import check_X_y

import numpy

import jax.numpy as np
from jax.scipy.linalg import cholesky, cho_solve, solve
from jax.scipy.special import expit
from jax import ops
from jax import jit


@partial(jit, static_argnums=0)
def _newton_iteration(y_train, K, f):
    pi = expit(f)
    W = pi * (1 - pi)
    # Line 5
    W_sr = np.sqrt(W)
    W_sr_K = W_sr[:, np.newaxis] * K
    B = np.eye(W.shape[0]) + W_sr_K * W_sr
    L = cholesky(B, lower=True)
    # Line 6
    b = W * f + (y_train - pi)
    # Line 7
    a = b - W_sr * cho_solve((L, True), W_sr_K.dot(b))
    # Line 8
    f = K.dot(a)

    # Line 10: Compute log marginal likelihood in loop and use as
    #          convergence criterion
    lml = -0.5 * a.T.dot(f) \
        - np.log1p(np.exp(-(y_train * 2 - 1) * f)).sum() \
        - np.log(np.diag(L)).sum()
    return lml, f, (pi, W_sr, L, b, a)


class BinaryGaussianProcessClassifier(_BinaryGaussianProcessClassifierLaplace):
    def _posterior_mode(self, K, return_temporaries=False):
        """Mode-finding for binary Laplace GPC and fixed kernel.
        This approximates the posterior of the latent function values for given
        inputs and target observations with a Gaussian approximation and uses
        Newton's iteration to find the mode of this approximation.
        """
        # Based on Algorithm 3.1 of GPML

        # If warm_start are enabled, we reuse the last solution for the
        # posterior mode as initialization; otherwise, we initialize with 0
        if self.warm_start and hasattr(self, "f_cached") \
           and self.f_cached.shape == self.y_train_.shape:
            f = self.f_cached
        else:
            f = np.zeros_like(self.y_train_, dtype=np.float32)

        # Use Newton's iteration method to find mode of Laplace approximation
        log_marginal_likelihood = -np.inf
        newton_iteration = partial(_newton_iteration, self.y_train_, K)

        for _ in range(self.max_iter_predict):
            lml, f, (pi, W_sr, L, b, a) = newton_iteration(f)
            # Check if we have converged (log marginal likelihood does
            # not decrease)
            # XXX: more complex convergence criterion
            if lml - log_marginal_likelihood < 1e-10:
                break
            log_marginal_likelihood = lml

        self.f_cached = f  # Remember solution for later warm-starts
        if return_temporaries:
            return log_marginal_likelihood, (pi, W_sr, L, b, a)
        else:
            return log_marginal_likelihood

    def log_marginal_likelihood(self, theta=None, eval_gradient=False,
                                clone_kernel=False):
        """Returns log-marginal likelihood of theta for training data.

        Parameters
        ----------
        theta : array-like of shape (n_kernel_params,) or None
            Kernel hyperparameters for which the log-marginal likelihood is
            evaluated. If None, the precomputed log_marginal_likelihood
            of ``self.kernel_.theta`` is returned.
        eval_gradient : bool, default: False
            If True, the gradient of the log-marginal likelihood with respect
            to the kernel hyperparameters at position theta is returned
            additionally. If True, theta must not be None.
        clone_kernel : bool, default=True
            If True, the kernel attribute is copied. If False, the kernel
            attribute is modified, but may result in a performance improvement.
        Returns
        -------
        log_likelihood : float
            Log-marginal likelihood of theta for training data.
        log_likelihood_gradient : array, shape = (n_kernel_params,), optional
            Gradient of the log-marginal likelihood with respect to the kernel
            hyperparameters at position theta.
            Only returned when eval_gradient is True.
        """

        if theta is None:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated for theta!=None")
            return self.log_marginal_likelihood_value_

        kernel_matrix_fn = self.kernel_.get_kernel_matrix_fn(eval_gradient)

        if eval_gradient:
            K, K_gradient = kernel_matrix_fn(theta, self.X_train_, None)
        else:
            K = kernel_matrix_fn(theta, self.X_train_, None)

        # Compute log-marginal-likelihood Z and also store some temporaries
        # which can be reused for computing Z's gradient
        Z, (pi, W_sr, L, b, a) = \
            self._posterior_mode(K, return_temporaries=True)

        if not eval_gradient:
            return Z

        # Compute gradient based on Algorithm 5.1 of GPML

        d_Z = np.empty(theta.shape[0])
        # XXX: Get rid of the np.diag() in the next line
        R = W_sr[:, np.newaxis] * cho_solve((L, True), np.diag(W_sr))  # Line 7
        C = solve(L, W_sr[:, np.newaxis] * K)  # Line 8
        # Line 9: (use einsum to compute np.diag(C.T.dot(C))))
        s_2 = -0.5 * (np.diag(K) - np.einsum('ij, ij -> j', C, C)) \
            * (pi * (1 - pi) * (1 - 2 * pi))  # third derivative

        for j in range(d_Z.shape[0]):
            C = K_gradient[:, :, j]   # Line 11
            # Line 12: (R.T.ravel().dot(C.ravel()) = np.trace(R.dot(C)))
            s_1 = .5 * a.T.dot(C).dot(a) - .5 * R.T.ravel().dot(C.ravel())

            b = C.dot(self.y_train_ - pi)  # Line 13
            s_3 = b - K.dot(R.dot(b))  # Line 14

            d_Z = ops.index_update(d_Z, j, s_1 + s_2.T.dot(s_3))  # Line 15

        return (
            numpy.asarray(Z, dtype=numpy.float64),
            numpy.asarray(d_Z, dtype=numpy.float64)
        )


class GaussianProcessClassifier(GPC):
    def fit(self, X, y):
        """Fit Gaussian process classification model
        Parameters
        ----------
        X : sequence of length n_samples
            Feature vectors or other representations of training data.
            Could either be array-like with shape = (n_samples, n_features)
            or a list of objects.
        y : array-like of shape (n_samples,)
            Target values, must be binary
        Returns
        -------
        self : returns an instance of self.
        """
        if self.kernel is None or self.kernel.requires_vector_input:
            X, y = check_X_y(X, y, multi_output=False,
                             ensure_2d=True, dtype="numeric")
        else:
            X, y = check_X_y(X, y, multi_output=False,
                             ensure_2d=False, dtype=None)

        self.base_estimator_ = BinaryGaussianProcessClassifier(
            self.kernel, self.optimizer, self.n_restarts_optimizer,
            self.max_iter_predict, self.warm_start, self.copy_X_train,
            self.random_state)

        self.classes_ = numpy.unique(y)
        self.n_classes_ = self.classes_.size
        if self.n_classes_ == 1:
            raise ValueError("GaussianProcessClassifier requires 2 or more "
                             "distinct classes; got %d class (only class %s "
                             "is present)"
                             % (self.n_classes_, self.classes_[0]))
        if self.n_classes_ > 2:
            if self.multi_class == "one_vs_rest":
                self.base_estimator_ = \
                    OneVsRestClassifier(self.base_estimator_,
                                        n_jobs=self.n_jobs)
            elif self.multi_class == "one_vs_one":
                self.base_estimator_ = \
                    OneVsOneClassifier(self.base_estimator_,
                                       n_jobs=self.n_jobs)
            else:
                raise ValueError("Unknown multi-class mode %s"
                                 % self.multi_class)

        self.base_estimator_.fit(X, y)

        if self.n_classes_ > 2:
            self.log_marginal_likelihood_value_ = numpy.mean(
                [estimator.log_marginal_likelihood()
                 for estimator in self.base_estimator_.estimators_])
        else:
            self.log_marginal_likelihood_value_ = \
                self.base_estimator_.log_marginal_likelihood()

        return self
