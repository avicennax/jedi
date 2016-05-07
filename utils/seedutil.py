# seed.py
# Various environment helper functions for random
# seeds, including saving and loading seeds.

from __future__ import division
import os.path
import datetime

from numpy import sqrt
import numpy as np

def save_seeds(seeds, description='', filename=None, dir='../data'):
    t = datetime.datetime.now()
    if filename is None:
        ts = ['seeds', t.year, t.month, t.day, t.hour, t.second]
        filename = '_'.join([str(t) for t in ts])
    path = os.path.join(dir, filename)
    np.save(path, seeds)

def load_seeds(filename, dir='../data'):
    path = os.path.join(dir, filename)
    return np.load(path)


def set_simulation_parameters(seed, N, i):
    """
    Common simulation parameters for FORCE/DFORCE

    seed: float
    N: int
    i: int

    """
    prng = np.random.RandomState(seed)

    J = prng.normal(0, sqrt(1 / N), (N, N)) # primary stochastic matrix
    Wi = prng.normal(0, sqrt(1 / i), (i, i)) # secondary stochastic matrix
    wi = prng.normal(0, sqrt(1 / i), (i, i)) # t
    x0 = prng.uniform(-0.5, 0.5, N) # initial x0

    u = prng.uniform(-1, 1, N)
    w = prng.uniform(-1 / sqrt(N), 1 / sqrt(N), N)  # Initial weights

    return J, Wi, wi, x0, u, w