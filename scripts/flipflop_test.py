from __future__ import division
import jedi
from utils import plot, seedutil

import matplotlib.pylab as plt
import numpy as np
from numpy import zeros,ones,eye,tanh,dot,outer,sqrt,linspace, \
    cos,pi,hstack,zeros_like,abs,repeat
from numpy.random import uniform,normal,choice

# Setting Seeds
seeds = uniform(0,10000,1).astype(int)

#
targets = np.load("../data/stability/flipflop/targets_tmax10sec.npy")
inputs = np.load("../data/stability/flipflop/inputs_tmax10sec.npy")

#Simulation parameters for FORCE
dt = .01      # time step
tmax = 10  # simulation length
tstop = 5  # learning stop time
g = 1.5    # gain factor?
N = 100      # size of stochastic pool
lr = 1   # learning rate
I = 1
rho = 100

errors = []
wus = []
zs = []

for seed in seeds:
    J, Wz, Wi, x0, u, w = seedutil.set_simulation_parameters(seed, N, I, p=(.1,1,1))

    def model(t0, x, params):
        z = params['z']
        tanh_x = params['tanh_x']
        inp = params['inp']
        return (-x + dot(J, tanh_x) + dot(Wi, inp) + Wz*z)/dt

    x,t,z,_,wu,_ = jedi.sforce(rho, targets, model, lr, dt, tmax, tstop, x0, w, inputs)

    zs.append(z)
    wus.append(wu)

    error = np.abs(z-np.array(targets))
    errors.append(error)

errors = np.array(errors)

# Figure 1
plt.figure(figsize=(12,5))
plot.target_vs_output_plus_error(t, z, wu, targets, offset=0, log=False)
plt.draw()

# Figure 2
plt.figure(figsize=(12,5))
plot.signal_error(errors, t, tstop, title= "FORCE (Flip Flop)", burn_in=5)
plt.draw()

plt.show()
