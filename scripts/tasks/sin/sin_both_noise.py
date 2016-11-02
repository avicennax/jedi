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
parameters['t_count'] = t_count = int(tmax/dt+2) # number of time steps
parameters['noise_int_var'] = noise_int_var = .3
parameters['noise_ext_var'] = noise_ext_var = .1

#Noise matrix
int_noise_mat = np.array([np.random.normal(0, noise_int_var, N) for _ in range(t_count)])
ext_noise_mat = np.random.normal(0, noise_ext_var, t_count)

targets = target(np.linspace(0, 10, t_count)) + ext_noise_mat

errors_noise = []
derrors_noise = []
zs_noise = []
dzs_noise = []

for seedling in seeds[:10]:
    J, Wz, _, x0, u, w = init_tools.set_simulation_parameters(seedling, N, 1, pE=pE, p=sparsity, rho=rho)

    # inp & z are dummy variables
    def model(t0, x, params):
        index = params['index']
        tanh_x = params['tanh_x']
        z = params['z']
        noise = params['noise'][index]
        return (-x + np.dot(J, tanh_x) + Wz*z + noise)/dt

    x, t, z, _, wu,_ = jedi.force(targets, model, lr, dt, tmax, tstart, tstop, x0, w, noise=int_noise_mat)

    zs_noise.append(z)
    error = z-targets
    errors_noise.append(error)

    x, t, z, _, wu,_ = jedi.dforce(jedi.step_decode, targets, model, lr, dt, tmax, tstart, tstop, x0, w,
                                   noise=int_noise_mat, pE=pE)
    dzs_noise.append(z)
    derror = z-targets
    derrors_noise.append(derror)

noise_errors = {}
noise_errors['parameters'] = parameters
noise_errors['force'] = (errors_noise, zs_noise)
noise_errors['dforce'] = (derrors_noise, dzs_noise)

noise_key = "({0}_{1})".format(noise_ext_var, noise_int_var)

cPickle.dump(noise_errors,
             open("../../../data/stability/sin/both_noise/noise_exin_" + noise_key + ".p", "wb"))