from __future__ import division
import jedi.jedi as jedi
from jedi.utils import plot, init_tools, seedutil
import cPickle

import matplotlib.pylab as plt
import numpy as np

# Setting Seeds
seeds = seedutil.load_seeds("main_seeds.npy", "../../../data/stability")

# sine-wave target
target = lambda t0: np.cos(2 * np.pi * t0/.5)

#Simulation parameters for FORCE
parameters = {}
parameters['dt'] = dt =.01      # time step
parameters['tmax'] = tmax = 10   # simulation length
parameters['tstop'] = tstop = 5 # learning stop time
parameters['tstart'] = tstart = 0 # learning start
parameters['N'] = N = 300      # size of stochastic pool
parameters['lr'] = lr = 1   # learning rate
parameters['rho'] = rho = 1.25 # spectral radius of J
parameters['pE'] = pE = .8 # excitatory percent
parameters['sparsity'] = sparsity = (.1,1,1) # weight sparsity

errors = []
derrors = []
zs = []
dzs = []

for seedling in seeds:
    J, Wz, _, x0, u, w = init_tools.set_simulation_parameters(seedling, N, 1, pE=pE, p=sparsity, rho=rho)

    # inp & z are dummy variables
    def model(t0, x, params):
        tanh_x = params['tanh_x']
        z = params['z']
        return (-x + np.dot(J, tanh_x) + Wz*z)/dt

    x, t, z, _, wu,_ = jedi.force(target, model, lr, dt, tmax, tstart, tstop, x0, w)

    zs.append(z)
    error = z-target(t)
    errors.append(error)

    x, t, z, _, wu,_ = jedi.dforce(jedi.step_decode, target, model, lr, dt, tmax, tstart, tstop, x0, w,
                                 pE=pE)

    dzs.append(z)
    derror = z-target(t)
    derrors.append(derror)

noiseless_errors = {}
noiseless_errors['parameters'] = parameters
noiseless_errors['force'] = (errors, zs)
noiseless_errors['dforce'] = (derrors, dzs)
cPickle.dump(noiseless_errors, open("../../../data/stability/sin/base/base.p", "wb"))
