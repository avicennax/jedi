# init_tools.py
# Tools for initializing weight matrices

from __future__ import division

from numpy import sqrt
import numpy as np
import scipy.stats as stats
from numpy.linalg import eigvals


def generate_ei(N, pE):
    """
    E/I signature.

    Parameters
    ----------
    N : int
        Number of recurrent units.
    pE : float, optional
         Fraction of units that are excitatory. Default is the usual value for cortex.

    https://github.com/voytekresearch/pycog
    """
    if pE > 1 or pE < 0:
        raise ValueError("pE must be := 0 < pE < 1.")

    Nexc = int(pE*N)

    idx = range(N)
    exc = idx[:Nexc]
    inh = idx[Nexc:]

    ei  = np.ones(N, dtype=int)
    ei[inh] *= -1

    # So the inhibitory neurons aren't overpowered by excititory ones.
    weights = np.ones(N)
    weights[exc] *= (1 - pE)
    weights[inh] *= pE

    return ei, weights


def set_simulation_parameters(seed, N, i, pE=None, p=None, rho=None):
    """
    Common simulation parameters for FORCE/DFORCE

    Parameters
    ----------
    seed : float
        PRNG seed
    N : int
        Size of primary stochastic matrix
    i : int
        Size of secondary stochastic matrix
    pE : float, optional
         Fraction of units that are excitatory. Default is the usual value for cortex.
    p : ndarray
        Sparsity mask parameters.
        Each p must be between [0,1].
    rho : float
        Spectral radius of recurrent weight matrix
    """
    prng = np.random.RandomState(seed)

    if p is None:
        p = np.ones(3)

    J = prng.normal(0, sqrt(1/ (p[0]*N)), (N, N)) # primary stochastic matrix
    Wz = prng.uniform(-1, 1, N) # feedback vector
    Wi = prng.normal(0, sqrt(1 / (p[2]*i)), (i, i)) # secondary stochastic matrix

    # sparsifying
    J *= stats.bernoulli.rvs(p[0], 0, N*N).reshape(N,N)
    Wz *= stats.bernoulli.rvs(p[1], 0, N)
    Wi *= stats.bernoulli.rvs(p[2], 0, i*i).reshape(i,i)

    x0 = prng.uniform(-0.5, 0.5, N) # initial state (x0)

    u = prng.uniform(-1, 1, N) #
    w = prng.uniform(-1 / sqrt(p[1]*N), 1 / sqrt(p[1]*N), N)  # Initial weights

    if pE is not None:
        ei, weights = generate_ei(N, pE)
        J = abs(J)
        J *= ei
        J *= weights
    if rho is not None:
        J *= rho/np.max(np.abs(eigvals(J)))


    return J, Wz, Wi, x0, u, w