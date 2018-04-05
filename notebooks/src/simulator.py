"""
"""

import numpy as np
import matplotlib.pyplot as plt


def simulate_frequencies(p, n_e, max_gen):
    """Simulate frequencies under the Wright Fisher model
    
    Args
    ----
    p : int
        number of SNPs
    n_e : int
        effective population size
    max_gen : int
        maximum number of generations
    
    Returns
    -------
    F : np.array
        max_gen x p matrix
    
    """
    F = np.empty((max_gen, p))
    F[0, :] = np.random.beta(1., 1., size=p)
    for t in range(1, max_gen):
        F[t, :] = np.random.binomial(2 * n_e, F[t-1, :]) / (2 * n_e)
    
    return(F)


def simulate_genotypes(t, F):
    """Simulate sampled genotypes conditional on 
    full allele frequency trajectories
    
    Args
    ----
    t : np.array
        sorted vector of sampled time points
    F : np.array
        max_gen x p matrix of allele frequency trajectories
    
    Returns
    -------
    X : np.array
        n x p genotype matrix
    """
    F_samp = F[t, :]
    X = np.random.binomial(2, F_samp)
    
    return(X)


def simulate_reads(X, eps, lamb):
    """Simulate read data conditional on the genotypes
    
    Args
    ----
    X : np.array
        n x p genotype matrix
    eps : float
        error probability
    lamb : float
        average coverage 
    
    Returns
    -------
    (Y, C) : tuple
        a tuple of np.arrays where Y is the read count of 
        the counted allele and C is the coverage 
    """
    n, p = X.shape
    C = np.random.poisson(lamb, size=(n, p)) 
    P = X / 2.
    Y = np.random.binomial(C, (eps * P) + ((1. - eps ) * (1. - P)))
    
    return((Y, C))


def est_het_geno(X):
    """Comptupes an estimate of mean heterozygosity from 
    the genotype data
    
    Args
    ----
    X : np.array
        n x p genotype matrix
    
    Returns
    -------
    h_hat : np.array
        estimate of mean heterozygosity for each sampled time 
        point
    """
    n, p = X.shape
    h_hat = np.sum(X==1, axis=1) / p
    
    return(h_hat)  


def est_het_reads(Y, C, eps):
    """
    """
    pass


def plot_xy(x, y, e=.005):
    """Plot xy scatter plot with y=x line
    
    Args
    ----
    x : np.array
        n x 1 array of values
    y : np.array
        n x 1 array of values
    e : float
        jitter to add at end of axis
    """
    x_min, x_max = (x.min(), x.max())
    y_min, y_max = (y.min(), y.max())
    plt.scatter(x, y)
    plt.xlim(x_min-e, x_max+e)
    plt.ylim(x_min-e, x_max+e)
    plt.plot([(x_min-e, y_min-e), (x_max+e, y_max+e)], 
             [(x_min-e, y_min-e), (x_max+e, y_max+e)], 
             'k-', alpha=0.75, zorder=0)
    
 