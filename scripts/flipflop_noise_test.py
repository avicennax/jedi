from __future__ import division
import jedi.jedi as jedi
from jedi.utils import plot, init_tools

import numpy as np
import cPickle

# Setting Seeds
seeds = np.random.uniform(0,10000,15).astype(int)

#
targets = np.load("../data/stability/flipflop/targets_tmax10sec.npy")
inputs = np.load("../data/stability/flipflop/inputs_tmax10sec.npy")

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
        z = params['z']
        tanh_x = params['tanh_x']
        inp = params['inputs']
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
cPickle.dump(noiseless_errors, open("../data/randompickles/noiseless_errors_ff.p", "wb"))

noise_errors={}

for noise_var in [.1,.2,.3,.4,.5]:

    #Noise matrix
    noise_mat = np.array([np.random.normal(0,noise_var,N) for i in range(t_count)])

    errors_noise = []
    derrors_noise = []
    zs_noise = []
    dzs_noise = []

    for seedling in seeds:
        J, Wz, Wi, x0, u, w = init_tools.set_simulation_parameters(seedling, N, I, pE=pE, p=sparsity, rho=rho)

        def model(t0, x, params):
            z = params['z']
            tanh_x = params['tanh_x']
            inp = params['inputs']
            noise = params['noise']
            return (-x + np.dot(J, tanh_x) + np.dot(Wi, inp) + Wz*z + noise)/dt

        x, t, z, _, wu,_ = jedi.force(targets, model, lr, dt, tmax, tstart, tstop, x0, w,
                                      inputs=inputs, noise=noise_mat)

        zs_noise.append(z)
        error = z-np.array(targets)
        errors_noise.append(error)

        x, t, z, _, wu,_ = jedi.dforce(jedi.step_decode, targets, model, lr, dt, tmax, tstart, tstop, x0, w,
                                 pE=pE, inputs=inputs, noise=noise_mat)

        dzs_noise.append(z)
        derror = z-np.array(targets)
        derrors_noise.append(derror)

    noise_errors[noise_var] = {}
    noise_errors[noise_var]['force'] = (errors_noise, zs_noise)
    noise_errors[noise_var]['dforce'] = (derrors_noise, dzs_noise)

cPickle.dump(noise_errors, open("../data/randompickles/noise_errors_ff.p", "wb"))