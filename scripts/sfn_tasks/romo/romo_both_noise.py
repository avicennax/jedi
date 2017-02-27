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
    parameters['noise_int_var'] = noise_int_var = .1
    parameters['noise_ext_var'] = noise_ext_var = .05
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

    I = 1
    noise_errors = {}
    noise_errors['parameters'] = parameters
    noise_errors['seeds'] = seeds

    rng = np.random.RandomState(seeds[0])
    parameters['trials'] = trials = [romo.generate_trial(rng, 5, params) for _
          in range(10)]

    for seedling in seeds:
        J, Wz, Wi, x0, u, w = init_tools.set_simulation_parameters(seedling, N, I, pE=pE, p=sparsity, rho=rho)

        def model(t0, x, params):
            index = params['index']
            z = params['z']
            tanh_x = params['tanh_x']
            inp = params['inputs'][index]
            return (-x + np.dot(J, tanh_x) + np.dot(Wi, inp) + Wz*z)/dt

        errors_noise = []
        derrors_noise = []
        zs_noise = []
        dzs_noise = []
        w_ = None

        for trial_num, trial in enumerate(trials):

            targets = trial['outputs'][:,1]
            inputs = trial['inputs'][:,1]
            tmax = float(len(targets))/100-.01

            #Noise matrix
            int_noise_mat = np.array([np.random.normal(0, noise_int_var, N) for i in range(len(targets))])
            ext_noise_mat = np.random.normal(0, noise_ext_var, len(targets))

            # Adding external noise
            inputs +=  ext_noise_mat

            if trial_num >= start_validate:
                tstop = .5*tmax
            else:
                tstop = tmax

            if w_ is not None:
                w = w_
                x0 = x[-1]

            x, t, z, _, wu,_ = jedi.force(targets, model, lr, dt, tmax, tstart, tstop, x0, w,
                                          inputs=inputs, noise=int_noise_mat)
            z = z[1:]
            zs_noise.append(z)
            error = z-np.array(targets)
            errors_noise.append(error)

            x, t, z, _, wu,_ = jedi.dforce(jedi.step_decode, targets, model, lr, dt, tmax, tstart, tstop, x0, w,
                                     pE=pE, inputs=inputs, noise=int_noise_mat)

            z = z[1:]
            dzs_noise.append(z)
            derror = z-np.array(targets)
            derrors_noise.append(derror)

        noise_errors[seedling] = {}
        noise_errors[seedling]['force'] = (errors_noise, zs_noise)
        noise_errors[seedling]['dforce'] = (derrors_noise, dzs_noise)

    noise_key = "({0}_{1})".format(noise_ext_var, noise_int_var)

    cPickle.dump(noise_errors,
                 open("../../../data/stability/romo/both_noise/noise_exin_" +
                      noise_key + ".p", "wb"))

if __name__ == "__main__":

    if len(sys.argv) > 1:
        try:
            seed = int(sys.argv[1])
        except:
            raise ValueError("Check your args; make sure the ONLY arg is an int")
    else:
        seed = None

    main(seed)
