#CHECK script parameters to ensure
# everything is koscher

from __future__ import division
from jedi import jedi
from jedi.utils import seedutil, mailer, init_tools

import sys
import time
import numpy as np
from numpy import tanh, dot, sqrt, cos, pi, abs
from scipy.integrate import odeint

def main(argv):
    # Loading seeds
    seeds = seedutil.load_seeds('main_seeds.npy', dir='../data/stability')

    ### Configuring Lorenz targets

    # Parameters specified by Abbott 2009.
    def lorentz((x, y, z), t0, sigma=10., beta=8./3, rho=28.0):
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

    break_in = 500
    T = 1501 # period
    x0 = np.random.randn(3) # starting vector
    t_= np.linspace(0, 60, T)
    lorenz = odeint(lorentz, x0, t_)/20
    targets = lorenz[break_in:,0]
    t_in = t_[break_in:]
    I = 1

    # Simulation parameters for FORCE
    parameters = {}
    parameters['dt'] = dt =.01      # time step
    parameters['tmax'] = tmax = 10   # simulation length
    parameters['tstop'] = tstop = 7 # learning stop time
    parameters['N'] = N = 1000     # size of stochastic pool
    parameters['lr'] = lr = 1   # learning rate
    parameters['rho'] = rho = 1.2 # spectral radius of J
    parameters['pE'] = pE = .8 # excitatory percent
    parameters['sparsity'] = sparsity = (.1,1,1) # weight sparsity

    xs = []
    zs = []
    wus = []

    # Script variables
    send_email = False
    ucsd_email = True
    checkpoint = 100
    run_dforce = True
    run_force = True
    write_params = True
    verbose = False

    if write_params:
        param_file = open('../data/stability/lorenz/parameters.txt', 'w')
        for key in parameters:
            param_file.write(": ".join([key, str(parameters[key])]))
            param_file.write('\n')
        param_file.close()

    # Seed subselection
    seeds = seeds[:100]

    if run_force:
        # Checkpoint timer
        timer = time.time()

        for seed_num, seedling in enumerate(seeds):

            J, Wz, _, x0, u, w = init_tools.set_simulation_parameters(seedling, N, I, pE=pE, p=sparsity, rho=rho)

            # inp & z are dummy variables
            def model(t0, x, tanh_x, inp, z):
                return (-x + dot(J, tanh_x) + Wz*z)/dt

            x, t, z, _, wu,_ = jedi.force(targets, model, lr, dt, tmax, tstop, x0, w, 0)

            xs.append(x)
            zs.append(z)
            wus.append(wu)

            if verbose:
                print "MC simulation %d complete." % seed_num

            if seed_num % checkpoint == 0 and seed_num != 0:
                xs = np.array(xs)
                zs = np.array(zs)
                wus = np.array(wus)

                np.save(''.join(['../data/stability/lorenz/x/FORCE_xs_',
                                str(seed_num-checkpoint+1), '-', str(seed_num)]), xs)
                np.save(''.join(['../data/stability/lorenz/z/FORCE_zs_',
                                str(seed_num-checkpoint+1), '-', str(seed_num)]), zs)
                np.save(''.join(['../data/stability/lorenz/wu/FORCE_wus_',
                                str(seed_num-checkpoint+1), '-', str(seed_num)]), wus)

                if send_email:
                    mailer.mail(argv, timer, seed_num, len(seeds), ucsd_email)
                timer = time.time()

                del xs, zs, wus
                xs = []
                zs = []
                wus = []

        if xs is not []:
            xs = np.array(xs)
            zs = np.array(zs)
            wus = np.array(wus)

            np.save(''.join(['../data/stability/lorenz/x/FORCE_xs_',
                                str(seed_num-len(xs)+1), '-', str(seed_num)]), xs)
            np.save(''.join(['../data/stability/lorenz/z/FORCE_zs_',
                                str(seed_num-len(xs)+1), '-', str(seed_num)]), zs)
            np.save(''.join(['../data/stability/lorenz/wu/FORCE_wus_',
                                str(seed_num-len(xs)+1), '-', str(seed_num)]), wus)

            if send_email:
                mailer.mail(argv, timer, seed_num, len(seeds), ucsd_email)

        np.save('../data/stability/lorenz/t.npy', t)

    if run_dforce:
        xs = []
        zs = []
        wus = []

        timer = time.time()

        for seed_num, seedling in enumerate(seeds):
            J, Wz, _, x0, u, w = init_tools.set_simulation_parameters(seedling, N, I, pE=pE, p=sparsity, rho=rho)

            # inp & z are dummy variables
            def model(t0, x, tanh_x, inp, z):
                return (-x + dot(J, tanh_x) + Wz*z)/dt

            x, t, z, _, wu,_ = jedi.dforce(jedi.step_decode, targets, model, lr, dt, tmax, tstop, x0, w, 0, pE=.8)

            xs.append(x)
            zs.append(z)
            wus.append(wu)

            if verbose:
                print "MC simulation %d complete." % seed_num

            if seed_num % checkpoint == 0 and seed_num != 0:
                xs = np.array(xs)
                zs = np.array(zs)
                wus = np.array(wus)

                np.save(''.join(['../data/stability/lorenz/x/DFORCE_xs_',
                                str(seed_num-checkpoint+1), '-', str(seed_num)]), xs)
                np.save(''.join(['../data/stability/lorenz/z/DFORCE_zs_',
                                str(seed_num-checkpoint+1), '-', str(seed_num)]), zs)
                np.save(''.join(['../data/stability/lorenz/wu/DFORCE_wus_',
                                str(seed_num-checkpoint+1), '-', str(seed_num)]), wus)

                if send_email:
                    mailer.mail(argv, timer, seed_num, len(seeds), ucsd_email)
                timer = time.time()

                del xs, zs, wus
                xs = []
                zs = []
                wus = []

        if xs is not []:
            xs = np.array(xs)
            zs = np.array(zs)
            wus = np.array(wus)

            np.save(''.join(['../data/stability/lorenz/x/DFORCE_xs_',
                                str(seed_num-len(xs)+1), '-', str(seed_num)]), xs)
            np.save(''.join(['../data/stability/lorenz/z/DFORCE_zs_',
                                str(seed_num-len(xs)+1), '-', str(seed_num)]), zs)
            np.save(''.join(['../data/stability/lorenz/wu/DFORCE_wus_',
                                str(seed_num-len(xs)+1), '-', str(seed_num)]), wus)

            if send_email:
                mailer.mail(argv, timer, seed_num, len(seeds), ucsd_email)


if __name__ == "__main__":
    main(sys.argv)