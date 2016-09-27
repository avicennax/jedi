from __future__ import division
import jedi.jedi as jedi
from jedi.utils import plot, init_tools

import random
import types
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.integrate import odeint, ode
from numpy import zeros,ones,eye,tanh,dot,outer,sqrt,linspace, \
    cos,pi,hstack,zeros_like,abs,repeat
from numpy.random import uniform,normal,choice

# Setting Seeds
seeds = uniform(0,10000,1).astype(int)

# sine-wave target
target = lambda t0: cos(2 * pi * t0/.5)

#Simulation parameters for FORCE
dt = .01      # time step
tmax = 10  # simulation length
tstop = 5  # learning stop time
rho = 1.25   # spectral radius of J
N = 300      # size of stochastic pool
lr = 1.0   # learning rate
pE = .8 # percent excitatory
sparsity = (.1,1,1) # sparsity
errors = []

for seedling in seeds:
    J, Wz, _, x0, u, w = init_tools.set_simulation_parameters(seedling, N, 1, pE=pE, p=sparsity, rho=rho)

    # inp & z are dummy variables
    def model(t0, x, tanh_x, inp, z):
        return (-x + dot(J, tanh_x) + Wz*z)/dt

    x,t,z,w_learn,wu,_ = jedi.force(target, model, lr, dt, tmax, tstop, x0, w, 0)

    error = np.abs(z-target(t))
    errors.append(error)

errors = np.array(errors)

# Figure 1
plt.figure(figsize=(12,5))
plot.target_vs_output_plus_error(t, z, wu, target, offset=1, log=False)
plt.draw()

# Figure 2
plt.figure(figsize=(12,5))
plot.signal_error(errors, t, tstop, title= "FORCE (Sin Wave)", burn_in=5)
plt.draw()

#
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
