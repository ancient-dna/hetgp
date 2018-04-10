"""Module for extensions to gpflow to allow for 
a Gaussian Process regression model with heteroskadastic
errors in the likelihood and a spatial-temporal kernel. Much of this
is modified from ...

https://github.com/GPflow/GPflow/blob/master/gpflow/models/gpr.py
"""
import tensorflow as tf
import gpflow
import numpy as np


class GPRHet(gpflow.models.GPModel):
    
    def __init__(self, X, Y, kern, mean_function=None, **kwargs):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        kern, mean_function are appropriate GPflow objects
        """
        likelihood = gpflow.likelihoods.Gaussian()
        X = gpflow.params.DataHolder(X)
        Y = gpflow.params.DataHolder(Y)
        gpflow.models.GPModel.__init__(self, X, Y, kern, likelihood, mean_function, **kwargs)
        self.num_data, self.input_dim = X.shape
        self.s2_hat = gpflow.Param(np.random.randn(self.num_data))

    @gpflow.decors.name_scope('likelihood')
    @gpflow.decors.params_as_tensors
    def _build_likelihood(self):
        """
        Construct a tensorflow function to compute the likelihood.
            \log p(Y | theta).
        """
        K = self.kern.K(self.X) + tf.diag(self.s2_hat) + (tf.eye(tf.shape(self.X)[0], dtype=gpflow.settings.float_type) * self.likelihood.variance)
        L = tf.cholesky(K)
        m = self.mean_function(self.X)
        logpdf = gpflow.densities.multivariate_normal(self.Y, m, L)  # (R,) log-likelihoods for each independent dimension of Y

        return tf.reduce_sum(logpdf)

    @gpflow.decors.name_scope('predict')
    @gpflow.decors.params_as_tensors
    def _build_predict(self, Xnew, full_cov=False):
        """
        Xnew is a data matrix, point at which we want to predict
        This method computes
            p(F* | Y )
        where F* are points on the GP at Xnew, Y are noisy observations at X.
        """
        Kx = self.kern.K(self.X, Xnew)
        K = self.kern.K(self.X) + tf.diag(self.s2_hat) + (tf.eye(tf.shape(self.X)[0], dtype=gpflow.settings.float_type) * self.likelihood.variance)
        L = tf.cholesky(K)
        A = tf.matrix_triangular_solve(L, Kx, lower=True)
        V = tf.matrix_triangular_solve(L, self.Y - self.mean_function(self.X))
        fmean = tf.matmul(A, V, transpose_a=True) + self.mean_function(Xnew)
        if full_cov:
            fvar = self.kern.K(Xnew) - tf.matmul(A, A, transpose_a=True)
            shape = tf.stack([1, 1, tf.shape(self.Y)[1]])
            fvar = tf.tile(tf.expand_dims(fvar, 2), shape)
        else:
            fvar = self.kern.Kdiag(Xnew) - tf.reduce_sum(tf.square(A), 0)
            fvar = tf.tile(tf.reshape(fvar, (-1, 1)), [1, tf.shape(self.Y)[1]])
        return fmean, fvar

    
class ExpNonSep(gpflow.kernels.Stationary):
    """
    A non-separable exponential kernel
    """
    @gpflow.decors.params_as_tensors
    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)

        r2 = self.scaled_square_dist(X=X, X2=X2)
        r = tf.sqrt(r2 + 1e-12)

        return self.variance * tf.exp(-.5 * r)
    
