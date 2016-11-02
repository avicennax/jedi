from __future__ import division
import jedi.jedi as jedi
from jedi.utils import plot, init_tools, seedutil

import numpy as np
import cPickle

# Setting Seeds
seeds = seedutil.load_seeds("main_seeds.npy", "../../../data/stability")

#
targets = np.load("../../../data/stability/flipflop/targets_tmax10sec.npy")
inputs = np.load("../../../data/stability/flipflop/inputs_tmax10sec.npy")

#Simulation parameters for FORCE
parameters = {}
parameters['dt'] = dt =.01      # time step
parameters['tmax'] = tmax = 10   # simulation length
parameters['tstop'] = tstop = 5 # learning stop time
parameters['tstart'] = tstart = 0 # learning start
parameters['N'] = N = 300      # size of stochastic pool
parameters['lr'] = lr = 1   # learning rate
parameters['rho'] = rho = 1.02 # spectral radius of J
parameters['pE'] = pE = .8 # excitatory percent
parameters['sparsity'] = sparsity = (.1,1,1) # weight sparsity
parameters['t_count'] = t_count = int(tmax/dt+2)

noiseless_errors = {}
errors = []
derrors = []
zs = []
dzs = []
I = 1

for seedling in seeds:
    J, Wz, Wi, x0, u, w = init_tools.set_simulation_parameters(seedling, N, I, pE=pE, p=sparsity, rho=rho)

    def model(t0, x, params):
        index = params['index']
        z = params['z']
        tanh_x = params['tanh_x']
        inp = params['inputs'][index]
        return (-x + np.dot(J, tanh_x) + np.dot(Wi, inp) + Wz*z)/dt

    x, t, z, _, wu,_ = jedi.force(targets, model, lr, dt, tmax, tstart, tstop, x0, w,
                                  inputs=inputs)

    zs.append(z)
    error = z-np.array(targets)
    errors.append(error)

    x, t, z, _, wu,_ = jedi.dforce(jedi.step_decode, targets, model, lr, dt, tmax, tstart, tstop, x0, w,
                                 pE=pE, inputs=inputs)

    dzs.append(z)
    derror = z-np.array(targets)
    derrors.append(derror)

noiseless_errors = {}
noiseless_errors['parameters'] = parameters
noiseless_errors['force'] = (errors, zs)
noiseless_errors['dforce'] = (derrors, dzs)
cPickle.dump(noiseless_errors, open("../../../data/stability/flipflop/base/base.p", "wb"))
