from __future__ import division
import jedi.jedi as jedi
from jedi.utils import plot, init_tools, seedutil
import examples.models.romo as romo
import pycog

import numpy as np
import cPickle
import sys
import time

def main(seed):

    # Setting Seeds
    seeds = seedutil.load_seeds("main_seeds.npy", "../../../data/stability")
    if seed is not None:
        seeds = seeds[:seed]

    #Simulation parameters for FORCE
    parameters = {}
    parameters['dt'] = dt = .01      # time step
    parameters['tstart'] = tstart = 0 # learning start
    parameters['N'] = N = 300      # size of stochastic pool
    parameters['lr'] = lr = 1   # learning rate
    parameters['rho'] = rho = 1.02 # spectral radius of J
    parameters['pE'] = pE = .8 # excitatory percent
    parameters['sparsity'] = sparsity = (.1,1,1) # weight sparsity
    parameters['trial_num'] = trial_num = 10
    parameters['start_validate'] = start_validate = 7

    # Romo task parameters
    params = {
        'callback_results': None,
        'target_output':    True,
        'minibatch_index':  1,
        'best_costs':       None,
        'name':             "gradient"
        }

    noiseless_errors = {}
    noiseless_errors['parameters'] = parameters

    rng = np.random.RandomState(seeds[0])
    parameters['trials'] = trials = [romo.generate_trial(rng, 10, params) for _
          in range(10)]

    for seedling in seeds:

        J, Wz, Wi, x0, u, w = init_tools.set_simulation_parameters(seedling, N, 1, pE=pE, p=sparsity, rho=rho)

        def model(t0, x, params):
            index = params['index']
            z = params['z']
            tanh_x = params['tanh_x']
            inp = params['inputs'][index]
            return (-x + np.dot(J, tanh_x) + np.dot(Wi, inp) + Wz*z)/dt

        errors = []
        derrors = []
        zs = []
        dzs = []
        w_ = None

        for trial_num, trial in enumerate(trials):

            targets = trial['outputs'][:,1]
            inputs = trial['inputs'][:,1]
            tmax = float(len(targets))/100-.01

            if trial_num >= start_validate:
                tstop = .5*tmax
            else:
                tstop = tmax

            if w_ is not None:
                w = w_
                x0 = x[-1]

            x, t, z, _, wu,_ = jedi.force(targets, model, lr, dt, tmax, tstart, tstop, x0, w,
                                          inputs=inputs)
            z = z[1:]
            zs.append(z)
            error = z-np.array(targets)
            errors.append(error)

            x, t, z, _, wu,_ = jedi.dforce(jedi.step_decode, targets, model, lr, dt, tmax, tstart, tstop, x0, w,
                                         pE=pE, inputs=inputs)
            z = z[1:]
            dzs.append(z)
            derror = z-np.array(targets)
            derrors.append(derror)

        noiseless_errors[seedling] = {}
        noiseless_errors[seedling]['force'] = (errors, zs)
        noiseless_errors[seedling]['dforce'] = (derrors, dzs)

    cPickle.dump(noiseless_errors, open("../../../data/stability/romo/base/base.p", "wb"))


if __name__ == "__main__":

    if len(sys.argv) > 1:
        try:
            seed = int(sys.argv[1])
        except:
            raise ValueError("Check your args; make sure the ONLY arg is an int")
    else:
        seed = None

    main(seed)

