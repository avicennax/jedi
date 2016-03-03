# jedi.py
# Exploring the discrete of FORCE
# Authors: Erik Peterson, Simon Haxby

from __future__ import division
import numpy as np
from scipy.integrate import ode
from numpy import eye, tanh, dot, outer, zeros_like


def force(target, model, lr, dt, tmax, tstop, x0, w):
    """
    Abbott's FORCE algorithm.

    Parameters
    ----------
        target: function
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
        x0: ndarray
            Initial model state.
        w: ndarray
            Initial weight vector to be fit.

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
    """

    # Simulation data: state, output, time, weight updates
    x, z, t, wu = [x0], [], [0], [0]

    # Running estimate of the inverse correlation matrix
    P = eye(len(x0))

    # Set up ode solver
    solver = ode(model)
    solver.set_initial_value(x0)

    # Integrate ODE, update weights, repeat
    while t[-1] < tmax:
        tanh_x = tanh(x[-1])  # cache
        z.append(dot(w, tanh_x))
        error = target(t[-1]) - z[-1]
        q = dot(P, tanh_x)
        c = lr / (1 + dot(q, tanh_x))
        P = P - c * outer(q, q)
        w = w + c * error * q

        # Stop leaning here
        if t[-1] > tstop:
            lr = 0

        wu.append(np.sum(np.abs(c * error * q)))

        solver.set_f_params(tanh_x, w)
        solver.integrate(solver.t + dt)
        x.append(solver.y)
        t.append(solver.t)

    # last update for readout neuron
    z.append(dot(w, tanh_x))

    x = np.array(x)
    t = np.array(t)

    return x, t, z, w, wu

def decode(x, rho):
    xd = zeros_like(x)

    xd[x > rho] = 1
    xd[x < -rho] = -1

    return xd

def dforce(rho, target, model, lr, dt, tmax, tstop, x0, w):
    """
    Peterson's DFORCE algorithm.
    A.K.A Abbott's FORCE with binary thresholding.

    Parameters?: target, f, dt, tmax, tstop, x0, w, P, lr

    Parameters
    ----------
        rho: float
            Binary threshold for decode.
        target: function
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
        x0: ndarray
            Initial model state.
        w: ndarray
            Initial weight vector to be fit.

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
    """

    # Simulation data: state, output, time, weight updates
    x, z, t, wu = [x0], [], [0], [0]

    # Running estimate of the inverse correlation matrix
    P = eye(len(x0))

    # Set up ode solver
    solver = ode(model)
    solver.set_initial_value(x0)


    # Integrate ode, update weights, repeat
    while t[-1] < tmax:
        tanh_x = tanh(x[-1])
        tanh_xd = decode(tanh_x, rho)  # BINARY CODE INTRODUCED HERE!
        z.append(dot(w, tanh_xd))
        error = target(t[-1]) - z[-1]
        q = dot(P, tanh_xd)
        c = lr / (1 + dot(q, tanh_xd))
        P = P - c * outer(q, q)
        w = w + c * error * q

        # Stop training time
        if t[-1] > tstop:
            lr = 0

        wu.append(np.sum(np.abs(c * error * q)))

        solver.set_f_params(tanh_x, w)
        solver.integrate(solver.t + dt)
        x.append(solver.y)
        t.append(solver.t)

    # last update for readout neuron
    z.append(dot(w, tanh_x))

    x = np.array(x)
    t = np.array(t)

    return x, t, z, w, wu
