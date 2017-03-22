from __future__ import division
import jedi.jedi as jedi
from jedi.utils import plot, init_tools, analysis

from functools import partial

import matplotlib.pylab as plt
import numpy as np

# Setting Seeds
seeds = np.random.uniform(0,10000,1).astype(int)

# sine-wave target
targets = np.load("../data/stability/flipflop/targets_tmax10sec.npy")
inputs = np.load("../data/stability/flipflop/inputs_tmax10sec.npy")

#Simulation parameters for FORCE

dt = .01      # time step
tmax = 10  # simulation length
tstart = 0
tstop = 8  # learning stop time
rho = 1.25   # spectral radius of J
N = 300      # size of stochastic pool
lr = 1.0   # learning rate
sparsity = (1,1,1) # sparsity

errors = []
zs = []
wus = []

for seedling in seeds:
    J, Wz, Wi, x0, u, w0 = init_tools.set_simulation_parameters(seedling, N, 1, p=sparsity, rho=rho)

    # inp & z are dummy variables
    def model(t0, x, params):
        index = params['index']
        z = params['z']
        tanh_x = params['tanh_x']
        inp = params['inputs'][index]
        return (-x + np.dot(J, tanh_x) + np.dot(Wi, inp) + Wz*z)/dt

    x, t, z, w, wu,_ = jedi.force(targets, model, lr, dt, tmax, tstart, tstop, x0, w0, inputs=inputs)

    zs.append(z)
    wus.append(wu)
    error = np.abs(z - np.array(targets))
    errors.append(error)


ind = 0
plot.target_vs_output_plus_error(t, zs[ind], wus[ind], targets, offset=0, log=False)

errors = np.array(errors)

# Fixed points

F = lambda x: -x + np.dot(J, np.tanh(x)) + Wz*np.dot(w, np.tanh(x))/dt
minima = analysis.fixed_points(F, x, 10)


## -- DFORCE -- ##

derrors = []
zs = []
wus = []

act_f = partial(jedi.sigmoid, 100)

for seedling in seeds:
    J, Wz, Wi, x0, u, w0 = init_tools.set_simulation_parameters(seedling, N, 1, p=sparsity, rho=rho)

    def model(t0, x, params):
        index = params['index']
        z = params['z']
        tanh_x = params['tanh_x']
        inp = params['inputs'][index]
        return (-x + np.dot(J, tanh_x) + np.dot(Wi, inp) + Wz*z)/dt

    x, t, z, w, wu, _ = jedi.dforce(act_f, targets, model, lr, dt, tmax, tstart, tstop, x0, w0, inputs=inputs)

    zs.append(z)
    wus.append(wu)
    derror = np.abs(z - np.array(targets))
    derrors.append(derror)

derrors = np.array(derrors)

F = lambda x: -x + np.dot(J, np.tanh(x)) + Wz*np.dot(w, np.tanh(x))/dt
minima = analysis.fixed_points(F, x, 10)