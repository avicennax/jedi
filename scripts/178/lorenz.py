from __future__ import division
import jedi.jedi as jedi
from jedi.utils import plot, init_tools, seedutil, analysis
from scipy.integrate import odeint
import matplotlib.pyplot as plt

import numpy as np
import cPickle
import sys

def main(seed):
    # Setting Seeds
    seeds = [42]

    # Flags
    compute_minima = False
    plots = True

    # Parameters specified by Abbott 2009.
    def lorentz((x, y, z), t0, sigma=10., beta=8./3, rho=28.0):
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

    # Defining Lorenz targets
    break_in = 500
    T = 1501  # period
    x0 = np.random.randn(3)  # starting vector
    t_ = np.linspace(0, 60, T)
    lorenz = odeint(lorentz, x0, t_) / 10
    targets = lorenz[break_in:, 0]

    # Simulation parameters for FORCE
    parameters = {}
    parameters['dt'] = dt =.01      # time step
    parameters['tmax'] = tmax = 10   # simulation length
    parameters['tstop'] = tstop = 7 # learning stop time
    parameters['tstart'] = tstart = 0 # learning start
    parameters['N'] = N = 1000     # size of stochastic pool
    parameters['lr'] = lr = 1   # learning rate
    parameters['rho'] = rho = 1.2 # spectral radius of J
    parameters['pE'] = pE = .8 # excitatory percent
    parameters['sparsity'] = sparsity = (.1,1,1) # weight sparsity

    errors = []
    zs = []

    for seedling in seeds:
        J, Wz, Wi, x0, u, w = init_tools.set_simulation_parameters(seedling, N, 1, p=sparsity, rho=rho)

        def model(t0, x, params):
            z = params['z']
            tanh_x = params['tanh_x']
            return (-x + np.dot(J, tanh_x) + Wz*z)/dt

        x, t, z, _, wu,_ = jedi.force(targets, model, lr, dt, tmax, tstart, tstop, x0, w)

        zs.append(z)
        error = np.abs(z[1:]-np.array(targets))
        errors.append(error)

    if plots:
        # Figure 1
        plt.figure(figsize=(12, 5))
        plot.target_vs_output_plus_error(t, z, wu, targets, offset=1, log=False)
        plt.draw()

        # Figure 2
        plt.figure(figsize=(12, 5))
        plot.signal_error(errors, t[1:], tstop, title="FORCE (Lorenz)", burn_in=5)
        plt.draw()

        plt.show()

    if compute_minima:
        F = lambda x: (-x + np.dot(J, np.tanh(x)) + Wz * np.dot(w, np.tanh(x))) / dt
        minima = analysis.fixed_points(F, x, 2)

    try:
        parameters['t'] = t
    except NameError:
        print "t was not defined; check seed args and script for errors"

    res = {}
    res['parameters'] = parameters
    res['force'] = (errors, zs)

    if compute_minima:
        res['minima'] = minima

    cPickle.dump(res, open("../../../data/178/lorenz.p", "wb"))


if __name__ == "__main__":

    if len(sys.argv) > 1:
        try:
            seed = int(sys.argv[1])
        except:
            raise ValueError("Check your args; make sure the ONLY arg is an int")
    else:
        seed = None

    main(seed)