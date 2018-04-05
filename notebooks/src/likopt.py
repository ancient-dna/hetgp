"""
"""
import numpy as np
import cvxpy as cvx
import scipy.stats as stats


def comp_lik_mat(y_i, c_i, eps):
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
    """Inverse of a 2 x 2 matrix
    
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


def comp_fish_info(L, pi_hat):
    """Compute the "observed" fisher information matrix 
    plugging in the mle
    
    Args
    ----
    L : np.array
        p x 3 matrix of likelihoods
    pi_hat : np.array
    """
    I = np.empty((2, 2))
    denom = np.square(L @ pi_hat)
    d_02 = L[:, 0] - L[:, 2]
    d_21 = L[:, 2] - L[:, 1]
    d_12 = L[:, 1] - L[:, 2]
    
    I[0, 0] = -np.sum(np.square(d_02) / denom)
    I[0, 1] = -np.sum((d_02 * d_21) / denom)
    I[1, 0] = -I[0, 1]
    I[1, 1] = -np.sum(np.square(d_12) / denom)
    
    return(I)
    
    
def comp_lik_var(I, p):
    """Compute asymptomatic variance for an individual given 
    their mle estimate of expected genotype frequencies. This is computed from 
    the marginal variance of the inverse of the observed fisher infromation i.e.
    the asymptomatic covariance matrix of the the three genotype frequencies
    
    Args
    ----
    I : np.array
        fisher information matrix
    p : float
        number of snps

    Returns
    -------
    sigma2 : float
        error variance in the likelihood
    """
    S = inv22(-I) * p 
    sigma2 = S[1, 1]
    
    return(sigma2)