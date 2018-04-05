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

    


    