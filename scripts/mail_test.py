from __future__ import division
from jedi import jedi
from jedi.utils import plot, seedutil
from voyutils import mailinator

import random
import types
import sys
import time
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

        error = np.abs(z[0]-target(t))
        errors.append(error)

    errors = np.array(errors)

if __name__ ==  "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Pass 'decryptor' password as script arg for smtp code: see Voytek whiteboard")
    try:
        wall_start = time.time()
        clock_start = time.clock()

        # script meat
        main()

        wall_time = wall_start - time.time()
        clock_time = clock_start - time.clock()
    except Exception as err:
        wall_time = wall_start - time.time()
        clock_time = clock_start - time.clock()

        err_str = ''.join(['ERROR: ', str(sys.exc_info()[0])])
        wall_time_str = ''.join(['Wall-time: ', str(wall_time)])
        clock_time_str = ''.join(['Clock-time: ', str(clock_time)])
        msg = "\n".join([err_str, wall_time_str, clock_time_str])
    else:
        str = 'Script executed; no errors.'
        wall_time_str = ''.join(['Wall-time: ', str(wall_time)])
        clock_time_str = ''.join(['Clock-time: ', str(clock_time)])
        msg = "\n".join([str, wall_time_str, clock_time_str])

    decryptor = sys.argv[1]

    if len(sys.argv) > 2:
        pw_filename = sys.argv[2]
    else:
        pw_filename = 'mail_pw.simon'

    #pw = mailinator.get_password(decryptor)
    msg = mailinator.add_header(sys.argv[0], msg)
    mailinator.gmail(msg, "sjh808ss$$")


