from __future__ import division
import jedi.jedi as jedi
from jedi.utils import plot, init_tools

import matplotlib.pylab as plt
import numpy as np

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
I = 1

# Noise matrix
noise_mat = np.array([np.random.normal(0,.3,N) for i in range(int(tmax/dt+2))])

errors = []

for seedling in seeds:
    J, Wz, Wi, x0, u, w = init_tools.set_simulation_parameters(seedling, N, I, pE=pE, p=sparsity, rho=rho)

    def model(t0, x, params):
        z = params['z']
        tanh_x = params['tanh_x']
        inp = params['inputs']
        noise = params['noise']
        return (-x + np.dot(J, tanh_x) + np.dot(Wi, inp) + Wz*z + noise)/dt

    x,t,z,_,wu,_ = jedi.force(targets, model, lr, dt, tmax, tstart, tstop, x0, w,
                                    noise=noise_mat, inputs=inputs)

    error = np.abs(z-np.array(targets))
    errors.append(error)

errors = np.array(errors)


plt.figure(figsize=(12,5))
plt.legend()
plt.subplot(2,1,2)
for i in range(20):
    plt.plot(t[:], x[:, i]);
plt.subplot(2,1,1)
plt.plot(t, np.array(z), lw=2, label="output")

plt.show()

## -- DFORCE -- ##

derrors = []

for seedling in seeds:
    J, Wz, Wi, x0, u, w = init_tools.set_simulation_parameters(seedling, N, 1, pE=pE, p=sparsity, rho=rho)

    def model(t0, x, params):
        z = params['z']
        tanh_x = params['tanh_x']
        inp = params['inputs']
        noise = params['noise']
        return (-x + np.dot(J, tanh_x) + np.dot(Wi, inp) + Wz*z + noise)/dt

    x,t,z,_,wu,_ = jedi.dforce(jedi.step_decode, targets, model, lr, dt, tmax, tstart, tstop, x0, w,
                                     noise=noise_mat, pE=.8, inputs=inputs)

    derror = np.abs(z-np.array(targets))
    derrors.append(derror)

derrors = np.array(derrors)


plt.figure(figsize=(12,5))
plt.legend()
plt.subplot(2,1,2)
for i in range(20):
    plt.plot(t[:], x[:, i]);
plt.subplot(2,1,1)
plt.plot(t, np.array(z), lw=2, label="output");

plt.show()

plt.figure(figsize=(12,4))
plot.cross_signal_error(errors, derrors, t, tstart, tstop,
                        title="FORCE vs SFORCE (Sin Wave))", burn_in=100)

plt.show()
