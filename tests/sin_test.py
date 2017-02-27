from __future__ import division
import jedi.jedi as jedi
from jedi.utils import plot, init_tools

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
rho = 1.25   # spectral radius of J
N = 300      # size of stochastic pool
lr = 1.0   # learning rate
pE = .8 # percent excitatory
sparsity = (.1,1,1) # sparsity

# Noise matrix
noise_mat = np.array([np.random.normal(0,.3,N) for i in range(int(tmax/dt+2))])


errors = []
zs = []
wus = []

for seedling in seeds:
    J, Wz, _, x0, u, w = init_tools.set_simulation_parameters(seedling, N, 1, pE=pE, p=sparsity, rho=rho)

    # inp & z are dummy variables
    def model(t0, x, params):
        index = params['index']
        tanh_x = params['tanh_x']
        z = params['z']
        noise = params['noise'][index]
        return (-x + np.dot(J, tanh_x) + Wz*z + noise)/dt

    x, t, z, _, wu,_ = jedi.force(target, model, lr, dt, tmax, tstart, tstop, x0, w, noise=noise_mat)

    zs.append(z)
    wus.append(wu)
    error = np.abs(z-target(t))
    errors.append(error)

errors = np.array(errors)

# Visualizing activities of first 20 neurons
T = 300
plt.figure(figsize=(12,4))
plt.subplot(211)
plt.title("Neuron Dynamics");
for i in range(10):
    plt.plot(t[:T], x[:T, i]);

plt.subplot(212)
for i in range(10):
    plt.plot(t[-T:], x[-T:, i]);
    plt.xlim(t[-T], t[-1]);

plt.show()

## -- DFORCE -- ##

derrors = []
zs = []
wus = []

for seedling in seeds:
    J, Wz, _, x0, u, w = init_tools.set_simulation_parameters(seedling, N, 1, pE=pE, p=sparsity, rho=rho)

    def model(t0, x, params):
        index = params['index']
        tanh_x = params['tanh_x']
        z = params['z']
        noise = params['noise'][index]
        return (-x + np.dot(J, tanh_x) + Wz*z + noise)/dt

    x, t, z, _, wu,_ = jedi.dforce(jedi.step_decode, target, model, lr, dt, tmax, tstart, tstop, x0, w,
                                   noise=noise_mat, pE=pE)

    zs.append(z)
    wus.append(wu)
    derror = np.abs(z-target(t))
    derrors.append(derror)

derrors = np.array(derrors)

# Visualizing activities of first 20 neurons
T = 300
plt.figure(figsize=(12,4))
plt.subplot(211)
plt.title("Neuron Dynamics");
for i in range(10):
    plt.plot(t[:T], x[:T, i]);

plt.subplot(212)
for i in range(10):
    plt.plot(t[-T:], x[-T:, i]);
    plt.xlim(t[-T], t[-1]);

plt.show()

plt.figure(figsize=(12,4))
plot.cross_signal_error(errors, derrors, t, tstart, tstop,
                        title="FORCE vs SFORCE (Sin Wave))", burn_in=100)

plt.show()