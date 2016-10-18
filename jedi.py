# jedi.py
# Exploring the discrete of FORCE
# Authors: Erik Peterson, Simon Haxby

from __future__ import division
import types
import time
import numpy as np
from scipy.integrate import ode
from numpy import eye, tanh, dot, outer, zeros, ceil

def step_decode(x):
    return .5*np.sign(x)+.5

def sigmoid(rho, x):
    return 1/(1+np.exp(-rho*x))

def force(target, model, lr, dt, tmax, tstart, tstop, x0, w,
          inputs=None, ode_solver=None, solver_params=None, verbose=True, noise=None):
    """
    Abbott's FORCE algorithm.

    Parameters
    ----------
        target: function / list
            Signal for network to learn.
        model: function
            ODE mass model.
        lr: float
            RLS learning rate.
        dt: float
            Simulation time-step.
        tmax: float
            Simulation time threshold.
        tstop: float
            Learning time threshold.
        tstart: float
            Learning time start.
        x0: ndarray
            Initial model state.
        w: ndarray
            Initial weight vector to be fit.
        inputs: ndarray (Optional)
            Model inputs at each time-step.
        target_func: bool
            Specifies whether the target is a function or ndarray.
        ode_solver: scipy.integrate.ode (Optional)
            Pre-initialized ODE solver.
            NOTE: if solver_params not passed, then ode_solver will
            be ignored and a new solver will be initialized.
        solver_params: dict (Optional)
            Contains state and time variables from a previous system.
            Should contain:
                'x': ndarray
                't': ndarray
        verbose: bool (Optional)
            Specifies whether to run simulation run time.
        noise: ndarray
            Model noise

    Returns
    -------
        x: ndarray
            States across duration of simulation.
        t: ndarray
            Times corresponding to readouts of 'x' 'w' and 'z'.
        z: ndarray
            Model readout across duration of simulation.
        w: ndarray
            Learned model weights across duration of simulation.
        wu: ndarray
            Weight changes across duration of simulations
            (as induced by RLS).
        solver: scipy.integrate.ode
            ODE solver used in FORCE
    """

    # Running estimate of the inverse correlation matrix
    P = eye(len(x0))

    # Set up ode solver
    if ode_solver is None:
        solver = ode(model)
        solver.set_initial_value(x0)

        # Simulation data: state, output, time, weight updates
        x, z, t, wu = [x0], [], [0], [0]
    else:
        if solver_params is None:
            solver = ode(model)
            solver.set_initial_value(x0)

            # Simulation data: state, output, time, weight updates
            x, z, t, wu = [x0], [], [0], [0]
        else:
            solver = ode_solver
            if 't' not in solver_params:
                t = [0]
                solver.t = 0
            else:
                t = solver_params['t']
            x = [solver_params['x'][-1]]
            wu, z = [], [0]


    if inputs is None:
        inputs = zeros(int(ceil(tmax/dt))+1).tolist()

    if isinstance(target, types.FunctionType):
        target_func = True
    else:
        target_func = False

    prev_tmax = t[-1]
    index = 0

    # For updating solver model parameters
    model_params = {}

    # Timing simulation
    start_time = time.clock()

    # Integrate ODE, update weights, repeat
    while t[-1] < tmax + prev_tmax:
        tanh_x = tanh(x[-1])  # cache
        z.append(dot(w, tanh_x))

        if t[-1] > tstop + prev_tmax or t[-1] < tstart + prev_tmax:
            wc = 0
        else:
            if target_func:
                error = target(t[-1]) - z[-1]
            else:
                error = target[index] - z[-1]

            q = dot(P, tanh_x)
            c = lr / (1 + dot(q, tanh_x))
            P = P - c * outer(q, q)
            w = w + c * error * q
            wc = np.sum(np.abs(c * error * q))

        wu.append(wc)

        model_params['tanh_x'] = tanh_x
        model_params['inputs'] = inputs[index]
        model_params['z'] = z[-1]
        if noise is not None:
            model_params['noise'] = noise[index]

        solver.set_f_params(model_params)
        solver.integrate(solver.t + dt)
        x.append(solver.y)
        t.append(solver.t)

        # Allows for next input/target to be processed.
        index += 1


    if verbose:
        print 'Simulation run-time (wall): %.3f seconds' % (time.clock() - start_time)

    # last update for readout neuron
    z.append(dot(w, tanh_x))

    x = np.array(x)
    t = np.array(t)

    return x, t, z, w, wu, solver


def dforce(decoder, target, model, lr, dt, tmax, tstart, tstop, x0, w,
           inputs=None, ode_solver=None, solver_params=None, verbose=True, pE=None,
           noise=None):
    """
    Peterson's DFORCE algorithm.
    A.K.A Abbott's FORCE with decoder.

    Parameters
    ----------
        decoder: function
            Decoder function.
        target: function / ndarray
            Signal for network to learn.
        model: function
            ODE mass model.
        lr: float
            RLS learning rate.
        dt: float
            Simulation time-step.
        tmax: float
            Simulation time threshold.
        tstop: float
            Learning time end.
        tstart: float
            Learning time start
        x0: ndarray
            Initial model state.
        w: ndarray
            Initial weight vector to be fit.
        inputs: ndarray (Optional)
            Model inputs at each time-step.
        target_func: bool
            Specifies whether the target is a function or ndarray.
        ode_solver: scipy.integrate.ode (Optional)
            Pre-initialized ODE solver.
            NOTE: if solver_params not passed, then ode_solver will
            be ignored and a new solver will be initialized.
        solver_params: dict (Optional)
            Contains state and time variables from a previous system.
            Should contain:
                'x': ndarray
                't': ndarray
        verbose: bool (Optional)
            Specifies whether to run simulation run time.
        pE: float
            Percent of J units that are excititory.
        noise: ndarray
            Model noise

    Returns
    -------
        x: ndarray
            States across duration of simulation.
        t: ndarray
            Times corresponding to readouts of 'x' 'w' and 'z'.
        z: ndarray
            Model readout across duration of simulation.
        w: ndarray
            Learned model weights across duration of simulation.
        wu: ndarray
            Weight changes across duration of simulations
            (as induced by RLS).
        solver: scipy.integrate.ode
            ODE solver used in DFORCE
    """


   # Running estimate of the inverse correlation matrix
    P = eye(len(x0))

    # Set up ode solver
    if ode_solver is None:
        solver = ode(model)
        solver.set_initial_value(x0)

        # Simulation data: state, output, time, weight updates
        x, z, t, wu = [x0], [], [0], [0]
    else:
        if solver_params is None:
            solver = ode(model)
            solver.set_initial_value(x0)

            # Simulation data: state, output, time, weight updates
            x, z, t, wu = [x0], [], [0], [0]
        else:
            solver = ode_solver
            if 't' not in solver_params:
                t = [0]
                solver.t = 0
            else:
                t = solver_params['t']
            x = [solver_params['x'][-1]]
            wu, z = [], [0]


    if inputs is None:
        inputs = zeros(int(ceil(tmax/dt))+1).tolist()

    if isinstance(target, types.FunctionType):
        target_func = True
    else:
        target_func = False

    prev_tmax = t[-1]
    index = 0

    # For updating solver model parameters
    model_params = {}

    # Timing simulation
    start_time = time.clock()

    # Integrate ODE, update weights, repeat
    while t[-1] < tmax + prev_tmax:

        tanh_x = tanh(x[-1])  # cache
        if pE is not None:
            e_count = int(pE*len(tanh_x))
            tanh_xd = np.concatenate([decoder(tanh_x[e_count:]), tanh_x[:e_count]])
        else:
            tanh_xd = decoder(tanh_x)
        z.append(dot(w, tanh_xd))

        # Stop leaning here
        if t[-1] > tstop + prev_tmax or t[-1] < tstart + prev_tmax:
            wc = 0
        else:
            if target_func:
                error = target(t[-1]) - z[-1]
            else:
                error = target[index] - z[-1]

            q = dot(P, tanh_xd)
            c = lr / (1 + dot(q, tanh_xd))
            P = P - c * outer(q, q)
            w = w + c * error * q
            wc = np.sum(np.abs(c * error * q))

        wu.append(wc)

        model_params['tanh_x'] = tanh_x
        model_params['inputs'] = inputs[index]
        model_params['z'] = z[-1]
        if noise is not None:
            model_params['noise'] = noise[index]

        solver.set_f_params(model_params)
        solver.integrate(solver.t + dt)
        x.append(solver.y)
        t.append(solver.t)

        # Allows for next input/target to be processed.
        index += 1

    if verbose:
        print 'Simulation run-time (wall): %.3f seconds' % (time.clock() - start_time)

    # last update for readout neuron
    z.append(dot(w, tanh_xd))

    x = np.array(x)
    t = np.array(t)

    return x, t, z, w, wu, solver