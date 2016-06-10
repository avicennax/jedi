from __future__ import division
import jedi
from utils import plot, seedutil

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

# lorenz target
# Parameters specified by Abbott 2009.
def lorentz((x, y, z), t0, sigma=10., beta=8./3, rho=28.0):
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

break_in = 5000
T = 15001 # period
x0 = np.random.randn(3) # starting vector
t_= np.linspace(0, 60, T)
lorenz = odeint(lorentz, x0, t_)/10
targets = lorenz[break_in:,0]

#Simulation parameters for FORCE
dt = .01      # time step
tmax = 100  # simulation length
tstop = 70  # learning stop time
g = 1.5    # gain factor?
N = 1000      # size of stochastic pool
lr = 1   # learning rate
errors = []

for seedling in seeds:
    J, Wz, _, x0, u, w = seedutil.set_simulation_parameters(seedling, N, 1, (.1,1,1))
    
    # inp & z are dummy variables
    def model(t0, x, tanh_x, inp, z): 
        return (-x + g * dot(J, tanh_x) + Wz*z)/dt
    
    x,t,z,_,wu,_ = jedi.force(targets, model, lr, dt, tmax, tstop, x0, w)

    error = np.abs(z-targets)
    errors.append(error)
    
errors = np.array(errors)

# Figure 1
plt.figure(figsize=(12,5))
plot.target_vs_output_plus_error(t, z, wu, targets, offset=0, log=False)
plt.draw()

# Figure 2
plt.figure(figsize=(12,5))
plot.signal_error(errors, t, tstop, title= "FORCE (Sin Wave)", burn_in=5)
plt.draw()

plt.show()
