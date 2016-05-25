from __future__ import division
from jedi import jedi
from jedi.utils import plot, seedutil

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
target = lambda t0: cos(2 * pi * t0 / 10)

#Simulation parameters for FORCE
dt = .1      # time step
tmax = 100   # simulation length
tstop = 50  # learning stop time
g = 1.5    # gain factor?
N = 300      # size of stochastic pool
lr = 1   # learning rate
errors = []

for seedling in seeds:
    J, Wz, _, x0, u, w = seedutil.set_simulation_parameters(seedling, N, 1, (.1,1,1))
    
    # inp & z are dummy variables
    def model(t0, x, tanh_x, inp, z): 
        return -x + g * dot(J, tanh_x) + Wz*z 
    
    x,t,z,_,wu,_ = jedi.force(target, model, lr, dt, tmax, tstop, x0, w)

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

plt.show()
