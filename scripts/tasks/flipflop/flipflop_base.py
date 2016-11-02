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
dt = .01      # time step
tmax = 10  # simulation length
tstart = 0
tstop = 5  # learning stop time
rho = 1.02   # spectral radius of J
N = 300      # size of stochastic pool
lr = 1.0   # learning rate
pE = .8 # percent excitatory
sparsity = (.1,1,1) # sparsity
I = 1 # input-dim
t_count = int(tmax/dt+2) # number of time steps

noiseless_errors = {}
errors = []
derrors = []
zs = []
dzs = []

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

noiseless_errors['force'] = (errors, zs)
noiseless_errors['dforce'] = (derrors, dzs)
cPickle.dump(noiseless_errors, open("../../../data/stability/flipflop/base/base.p", "wb"))
