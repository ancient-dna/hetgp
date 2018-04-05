"""A module for computing the maximum likelihood of 
the mixture propotions i.e. genotype frequencies and 
related qunatities.
"""
import numpy as np
import cvxpy as cvx
import scipy.stats as stats


def comp_lik_mat(y_i, c_i, eps, n_samp=0):
    """Compute likelihood matrix for an individual
    
    Args 
    ----
    y_i : np.array
        p x 1 vector of read counts of the counted allele
    c_i : np.array
        p x 1 vector of coverages 
    eps : float
        fixed error probability
        
    Returns
    -------
    L : np.array
        p x 3 likelihood matrix 
    """
    p = y_i.shape[0]
    K = 3
    L = np.empty((p, K))
    L[:,0] = stats.binom.pmf(y_i, c_i, eps, loc=0)
    L[:,1] = stats.binom.pmf(y_i, c_i, .5, loc=0)
    L[:,2] = stats.binom.pmf(y_i, c_i, 1. - eps, loc=0)
    
    if n_samp != 0:
        idx = np.random.choice(range(p), replace=True, size=n_samp)
        L = L[idx, :]
        
    return(L)
    
    
def est_freq_read(L):
    """Estimate expected genotype frequencies for an individual
    using maximum likelihood. This is a convex optimization 
    problem i.e. a mixture distribution with fixed components
    
    Args
    ----
    L : np.array
        p x 3 matrix of likelihoods
        
    Returns
    -------
    pi_hat : np.array
        3 x 1 vector of estimated mixture proportions
    """
    # variable for mixture proportions
    pi = cvx.Variable(3)
    
    # objective function
    objective = cvx.Minimize(cvx.neg(cvx.sum_entries(cvx.log(L * pi))))
    
    # constraints of mixture proportions
    constraints = [0 <= pi, pi <= 1, cvx.sum_entries(pi) == 1]
    
    # intailize the optimization problem
    prob = cvx.Problem(objective, constraints)
    
    # sovle the problem
    result = prob.solve()
    
    # the optimal value for the mixture proportions
    pi_hat = pi.value
    
    return(pi_hat)


def inv22(A):
    """Analytic Inverse of a 2 x 2 matrix
    
    Args
    ----
    A : np.array
        2 x 2 matrix
    
    Returns
    -------
    Ainv : np.array
        inverse of A
    """
    a = A[0, 0]
    b = A[0, 1]
    c = A[1, 0]
    d = A[1, 1]
    det = 1. / ((a * d) - (b * c))
    B = np.empty((2, 2))
    B[0, 0] = d
    B[0, 1] = -b
    B[1, 0] = -c
    B[1, 1] = a
    Ainv = det * B
    
    return(Ainv)


def comp_hessian(L, pi_hat):
    """Modified from ...
    
    https://github.com/pcarbo/mixopt/experiments/blob/master/code
    
    Computes the hessian of the negative log likelihood with respect to the 
    mixture proportions. We plugin the mle as our estimate for the mixture propotions
    in to hessian.
    
    Args
    ----
    L : np.array
        p x 3 matrix of likelihoods
        
    pi_hat : np.array
        3 x 1 vector of estimated mixture proportions
        
    Returns
    -------
    H : np.array
        3 x 3 hessian 
    """
    a = (1.0 / np.asarray(L @ pi_hat))
    Y = (a * L)
    H = Y.T @ Y
    
    return(H)
    
    
def comp_lik_var(L, pi_hat):
    """Computes the marginal variance of the likelihood by 
    computing the invsere of the hessian of the negative log-liklihood
    i.e. the negative fisher information matrix.
    
    Args
    ----
    L : np.array
        p x 3 matrix of likelihoods
        
    pi_hat : np.array
        3 x 1 vector of estimated mixture proportions
        
    Returns
    -------
    sigma2 : float
        error variance in the likelihood
    """
    H = comp_hessian(L, pi_hat)
    S = inv22(H)
    sigma2 = S[1, 1]
    
    return(sigma2)