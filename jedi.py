# jedi.py
# Exploring the discrete of FORCE
# Authors: Erik Peterson, Simon Haxby

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint,ode
from numpy import zeros, ones, eye, tanh, dot, outer, sqrt, \
    linspace, cos, pi, hstack, zeros_like, abs, repeat
from numpy.random import uniform, normal, choice
from collections import namedtuple

def force():
    """
    Abbott's FORCE algorithm.
    Parameters?: target, f, dt, tmax, tstop, x0, w, P, lr
    """
    target = lambda t0: cos(2 * pi * t0 / 50)  # target pattern

    f3 = lambda t0, x: -x + g * dot(J, tanh_x) + dot(w, tanh_x) * u

    dt = 1       # time step
    tmax = 800   # simulation length
    tstop = 300

    N = 300
    J = normal(0, sqrt(1 / N), (N, N))
    x0 = uniform(-0.5, 0.5, N)
    t = linspace(0, 50, 500)

    g = 1.5
    u = uniform(-1, 1, N)
    w = uniform(-1 / sqrt(N), 1 / sqrt(N), N)  # initial weights
    P = eye(N)  # Running estimate of the inverse correlation matrix
    lr = 1.0  # learning rate

    # simulation data: state, output, time, weight updates
    x, z, t, wu = [x0], [], [0], [0]

    # Set up ode solver
    solver = ode(f3)
    solver.set_initial_value(x0)

    # Integrate ode, update weights, repeat
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

        solver.integrate(solver.t + dt)
        x.append(solver.y)
        t.append(solver.t)

    # last update for readout neuron
    z.append(dot(w, tanh_x))

    x = np.array(x)
    t = np.array(t)

    return t, x, z, w, wu

def decode(x, rho):
    xd = zeros_like(x)

    xd[x > rho] = 1
    xd[x < -rho] = -1

    return xd

def dforce():
    """
    Peterson's DFORCE algorithm.
    A.K.A Abbott's FORCE with binary thresholding.

    Parameters?: target, f, dt, tmax, tstop, x0, w, P, lr
    """
    target = lambda t0: cos(2 * pi * t0 / 50)  # target pattern

    f3 = lambda t0, x: -x + g * dot(J, tanh_x) + dot(w, tanh_x) * u

    dt = 1       # time step
    tmax = 800   # simulation length
    tstop = 500

    N = 300
    J = normal(0, sqrt(1 / N), (N, N))
    x0 = uniform(-0.5, 0.5, N)

    g = 1.5
    u = uniform(-1, 1, N)
    w = uniform(-1 / sqrt(N), 1 / sqrt(N), N)  # initial weights
    P = eye(N)  # Running estimate of the inverse correlation matrix
    lr = .4  # learning rate

    rho = repeat(0.05, N)

    # simulation data: state,
    # output, time, weight updates
    x, z, t, wu = [x0], [], [0], [0]

    # Set up ode solver
    solver = ode(f3)
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

        solver.integrate(solver.t + dt)
        x.append(solver.y)
        t.append(solver.t)

    # last update for readout neuron
    z.append(dot(w, tanh_x))

    # plot
    x = np.array(x)
    t = np.array(t)

    return t, x, z, w, wu

def learning_plot(t, z, target, wu):
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(t, target(t), '-r', lw=2)
    plt.plot(t, z, '-b')
    plt.legend(('target', 'output'))
    plt.ylim([-1.1, 3])
    plt.xticks([])
    plt.subplot(2, 1, 2)
    plt.plot(t, wu, '-k')
    plt.yscale('log')
    plt.ylabel('$|\Delta w|$', fontsize=20)
    plt.xlabel('time', fontweight='bold', fontsize=16)
    plt.show()