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
parameters['t_count'] = t_count = int(tmax/dt+2) # number of time steps
parameters['noise_int_var'] = noise_int_var = .3
parameters['noise_ext_var'] = noise_ext_var = .1

errors = []
derrors = []
zs = []
dzs = []
I = 1

#Noise matrix
int_noise_mat = np.array([np.random.normal(0, noise_int_var, N) for i in range(t_count)])
ext_noise_mat = np.random.normal(0, noise_ext_var, t_count)

# Adding external noise
inputs +=  ext_noise_mat

errors_noise = []
derrors_noise = []
zs_noise = []
dzs_noise = []

for seedling in seeds:
    J, Wz, Wi, x0, u, w = init_tools.set_simulation_parameters(seedling, N, I, pE=pE, p=sparsity, rho=rho)

    def model(t0, x, params):
        index = params['index']
        z = params['z']
        tanh_x = params['tanh_x']
        inp = params['inputs'][index]
        noise = params['noise'][index]
        return (-x + np.dot(J, tanh_x) + np.dot(Wi, inp) + Wz*z + noise)/dt

    x, t, z, _, wu,_ = jedi.force(targets, model, lr, dt, tmax, tstart, tstop, x0, w,
                                  inputs=inputs, noise=int_noise_mat)

    zs_noise.append(z)
    error = z-np.array(targets)
    errors_noise.append(error)

    x, t, z, _, wu,_ = jedi.dforce(jedi.step_decode, targets, model, lr, dt, tmax, tstart, tstop, x0, w,
                             pE=pE, inputs=inputs, noise=int_noise_mat)

    dzs_noise.append(z)
    derror = z-np.array(targets)
    derrors_noise.append(derror)

noise_errors = {}
noise_errors['parameters'] = parameters
noise_errors['force'] = (errors_noise, zs_noise)
noise_errors['dforce'] = (derrors_noise, dzs_noise)

noise_key = "({0}_{1})".format(noise_ext_var, noise_int_var)

cPickle.dump(noise_errors,
             open("../../../data/stability/flipflop/both_noise/noise_exin_" + noise_key + ".p", "wb"))