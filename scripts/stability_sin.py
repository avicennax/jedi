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
    parameters = {}
    parameters['dt'] = dt =.1      # time step
    parameters['tmax'] = tmax = 100   # simulation length
    parameters['tstop'] = tstop = 50 # learning stop time
    parameters['g'] = g = 1.5    # gain factor?
    parameters['N'] = N = 300      # size of stochastic pool
    parameters['lr'] = lr = 1   # learning rate
    parameters['rho'] = rho = 100 # SFORCE sharpness factor

    xs = []
    zs = []
    wus = []

    checkpoint = 2

    param_file = open('../data/stability/sin/parameters.txt', 'w')
    for key in parameters:
        param_file.write(": ".join([key, str(parameters[key])]))
        param_file.write('\n')
    param_file.close()

    for seed_num, seedling in enumerate(seeds[:10]):
        J, Wz, _, x0, u, w = seedutil.set_simulation_parameters(seedling, N, 1, (.1,1,1))

        # inp & z are dummy variables
        def model(t0, x, tanh_x, inp, z):
            return -x + g * dot(J, tanh_x) + Wz*z

        x,t,z,_,wu,_ = jedi.force(target, model, lr, dt, tmax, tstop, x0, w)

        xs.append(x)
        zs.append(z)
        wus.append(wu)

        if seed_num % checkpoint == 0 and seed_num != 0:
            xs = np.array(xs)
            zs = np.array(zs)
            wus = np.array(wus)

            np.save(''.join(['../data/stability/sin/x/FORCE_xs_',
                            str(seed_num-checkpoint+1), '-', str(seed_num)]), xs)
            np.save(''.join(['../data/stability/sin/z/FORCE_zs_',
                            str(seed_num-checkpoint+1), '-', str(seed_num)]), zs)
            np.save(''.join(['../data/stability/sin/wu/FORCE_wus_',
                            str(seed_num-checkpoint+1), '-', str(seed_num)]), wus)

            del xs, zs, wus
            xs = []
            zs = []
            wus = []

    if xs is not []:
        xs = np.array(xs)
        zs = np.array(zs)
        wus = np.array(wus)

        np.save(''.join(['../data/stability/sin/x/FORCE_xs_',
                            str(seed_num-len(xs)+1), '-', str(seed_num)]), xs)
        np.save(''.join(['../data/stability/sin/z/FORCE_zs_',
                            str(seed_num-len(xs)+1), '-', str(seed_num)]), zs)
        np.save(''.join(['../data/stability/sin/wu/FORCE_wus_',
                            str(seed_num-len(xs)+1), '-', str(seed_num)]), wus)

    np.save('../data/stability/sin/t.npy', t)

    xs = []
    zs = []
    wus = []

    for seed_num, seedling in enumerate(seeds[:10]):
        J, Wz, _, x0, u, w = seedutil.set_simulation_parameters(seedling, N, 1, (.1,1,1))

        # inp & z are dummy variables
        def model(t0, x, tanh_x, inp, z):
            return -x + g * dot(J, tanh_x) + Wz*z

        x,t,z,_,wu,_ = jedi.sforce(rho, target, model, lr, dt, tmax, tstop, x0, w)

        xs.append(x)
        zs.append(z)
        wus.append(wu)

        if seed_num % checkpoint == 0 and seed_num != 0:
            xs = np.array(xs)
            zs = np.array(zs)
            wus = np.array(wus)

            np.save(''.join(['../data/stability/sin/x/DFORCE_xs_',
                            str(seed_num-checkpoint+1), '-', str(seed_num)]), xs)
            np.save(''.join(['../data/stability/sin/z/DFORCE_zs_',
                            str(seed_num-checkpoint+1), '-', str(seed_num)]), zs)
            np.save(''.join(['../data/stability/sin/wu/DFORCE_wus_',
                            str(seed_num-checkpoint+1), '-', str(seed_num)]), wus)

            del xs, zs, wus
            xs = []
            zs = []
            wus = []

    if xs is not []:
        xs = np.array(xs)
        zs = np.array(zs)
        wus = np.array(wus)

        np.save(''.join(['../data/stability/sin/x/DFORCE_xs_',
                            str(seed_num-len(xs)+1), '-', str(seed_num)]), xs)
        np.save(''.join(['../data/stability/sin/z/DFORCE_zs_',
                            str(seed_num-len(xs)+1), '-', str(seed_num)]), zs)
        np.save(''.join(['../data/stability/sin/wu/DFORCE_wus_',
                            str(seed_num-len(xs)+1), '-', str(seed_num)]), wus)


if __name__ ==  "__main__":
    main()