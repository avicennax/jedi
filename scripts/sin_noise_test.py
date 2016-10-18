from __future__ import division
import jedi.jedi as jedi
from jedi.utils import plot, init_tools
import cPickle

import matplotlib.pylab as plt
import numpy as np

# Setting Seeds
seeds = np.random.uniform(0,10000,15).astype(int)

# sine-wave target
target = lambda t0: np.cos(2 * np.pi * t0/.5)

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
t_count = int(tmax/dt+2) # number of time steps

noiseless_errors = {}
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

noiseless_errors['force'] = (errors, zs)
noiseless_errors['dforce'] = (derrors, dzs)

cPickle.dump(noiseless_errors, open("../data/randompickles/noiseless_errors.p", "wb"))

noise_errors={}

for noise_var in [.1,.2,.3,.4,.5]:

    #Noise matrix
    noise_mat = np.array([np.random.normal(0,noise_var,N) for i in range(t_count)])

    errors_noise = []
    derrors_noise = []
    zs_noise = []
    dzs_noise = []

    for seedling in seeds:
        J, Wz, _, x0, u, w = init_tools.set_simulation_parameters(seedling, N, 1, pE=pE, p=sparsity, rho=rho)

        # inp & z are dummy variables
        def model(t0, x, params):
            tanh_x = params['tanh_x']
            z = params['z']
            noise = params['noise']
            return (-x + np.dot(J, tanh_x) + Wz*z + noise)/dt

        x, t, z, _, wu,_ = jedi.force(target, model, lr, dt, tmax, tstart, tstop, x0, w, noise=noise_mat)

        zs_noise.append(z)
        error = z-target(t)
        errors_noise.append(error)

        x, t, z, _, wu,_ = jedi.dforce(jedi.step_decode, target, model, lr, dt, tmax, tstart, tstop, x0, w,
                                       noise=noise_mat, pE=pE)
        dzs_noise.append(z)
        derror = z-target(t)
        derrors_noise.append(derror)

    noise_errors[noise_var] = {}
    noise_errors[noise_var]['force'] = (errors_noise, zs_noise)
    noise_errors[noise_var]['dforce'] = (derrors_noise, dzs_noise)

cPickle.dump(noise_errors, open("../data/randompickles/noise_errors.p", "wb"))