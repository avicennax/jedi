from __future__ import division
import jedi.jedi as jedi
from jedi.utils import plot, init_tools, analysis

from functools import partial

import matplotlib.pylab as plt
import numpy as np

# Setting Seeds
seeds = np.random.uniform(0,10000,1).astype(int)

# sine-wave target
target = lambda t0: np.cos(2 * np.pi * t0/.5)

#Simulation parameters for FORCE

dt = .01      # time step
tmax = 10  # simulation length
tstart = 0
tstop = 5  # learning stop time
rho = 1.25  # spectral radius of J
N = 300      # size of stochastic pool
lr = 1.0   # learning rate
sparsity = (1,1,1) # sparsity

errors = []
zs = []
wus = []

for seedling in seeds:
    J, Wz, _, x0, u, w0 = init_tools.set_simulation_parameters(seedling, N, 1, p=sparsity, rho=rho)

    # inp & z are dummy variables
    def model(t0, x, params):
        index = params['index']
        tanh_x = params['tanh_x']
        z = params['z']
        return (-x + np.dot(J, tanh_x) + Wz*z)/dt

    x, t, z, w, wu,_ = jedi.force(target, model, lr, dt, tmax, tstart, tstop, x0, w0)

    zs.append(z)
    wus.append(wu)
    error = np.abs(z-target(t))
    errors.append(error)

errors = np.array(errors)

# Fixed points

F = lambda x: (-x + np.dot(J, np.tanh(x)) + Wz*np.dot(w, np.tanh(x)))/dt
minima = analysis.fixed_points(F, x, 10, plot=True)

