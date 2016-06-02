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
    # Loading seeds
    seeds = seedutil.load_seeds('main_seeds.npy', dir='../data/stability')

    # Sin wave target function
    target = lambda t0: cos(2 * pi * t0 / 10)
    # Simulation parameters for FORCE

    dt = .1      # time step
    tmax = 100   # simulation length
    tstop = 50 # learning stop time
    g = 1.5    # gain factor?
    N = 1000      # size of stochastic pool
    lr = 1   # learning rate
    rho = 100 # SFORCE sharpness factor

    xs = []
    zs = []
    wus = []

    for seedling in seeds:
        J, Wz, _, x0, u, w = seedutil.set_simulation_parameters(seedling, N, 1, (.1,1,1))

        # inp & z are dummy variables
        def model(t0, x, tanh_x, inp, z):
            return -x + g * dot(J, tanh_x) + Wz*z

        x,t,z,_,wu,_ = jedi.force(target, model, lr, dt, tmax, tstop, x0, w)

        xs.append(x)
        zs.append(z)
        wus.append(wu)

    np.save(xs, '../data/stability/sin/x/xs_FORCE_1000.npy')
    np.save(zs, '../data/stability/sin/z/zs_FORCE_1000.npy')
    np.save(wus, '../data/stability/sin/x/wus_FORCE_1000.npy')
    np.save(t, '../data/stability/sin/t_1000.npy')

    xs = []
    zs = []
    wus = []

    for seed in seeds:
        J, Wz, _, x0, u, w = seedutil.set_simulation_parameters(seed, N, 1, (.1,1,1))

        def model(t0, x, tanh_x, inp, z):
            return -x + g * dot(J, tanh_x) + Wz*z

        x,t,z,_,wu,_ = jedi.sforce(rho, target, model, lr, dt, tmax, tstop, x0, w)

        xs.append(x)
        zs.append(z)
        wus.append(wu)

    np.save(xs, '../data/stability/sin/x/xs_SFORCE_1000.npy')
    np.save(zs, '../data/stability/sin/z/zs_SFORCE_1000.npy')
    np.save(wus, '../data/stability/sin/x/wus_SFORCE_1000.npy')


if __name__ ==  "__main__":
    main()