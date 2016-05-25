from __future__ import division
from jedi import jedi
from jedi.utils import plot, seedutil

import random
import types
import sys
import numpy as np
from scipy.integrate import odeint, ode
from numpy import zeros,ones,eye,tanh,dot,outer,sqrt,linspace, \
    cos,pi,hstack,zeros_like,abs,repeat
from numpy.random import uniform,normal,choice

def main():
    # Setting Seeds
    seeds = uniform(0,10000,1).astype(int)

    # sine-wave target
    target = lambda t0: cos(2 * pi * t0 / 10)

    #Simulation parameters for FORCE
    dt = .1      # time step
    tmax = 100   # simulation length
    tstop = 50 # learning stop time
    g = 1.5    # gain factor?
    N = 300      # size of stochastic pool
    lr = 1   # learning rate
    rho = 100 # SFORCE sharpness factor
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

    derrors = []

    for seed in seeds:
        J, Wz, _, x0, u, w = seedutil.set_simulation_parameters(seed, N, 1, (.1,1,1))

        def model(t0, x, tanh_x, inp, z):
            return -x + g * dot(J, tanh_x) + Wz*z

        x,t,z,_,wu,_ = jedi.sforce(rho, target, model, lr, dt, tmax, tstop, x0, w)

        derror = np.abs(z-target(t))
        derrors.append(derror)

    derrors = np.array(derrors)


if __name__ ==  "__main__":
    main()