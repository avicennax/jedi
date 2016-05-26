# plot.py
# Various plotting functions for jedi
# Author: Simon Haxby

import numpy as np
import matplotlib.pyplot as plt
import types


def signal_error(errs, t, tstop, title, burn_in=0, mean=True):
    """
    errs: list[ndarray] (mean=True) / ndarray (mean=False)
    t: ndarray
    tstop: float
    title: string
    burn_in: float (optional)
    mean: bool (optional)
    """
    if mean:
        errs = np.mean(errs, axis=0)
    ymax = 2*np.max(errs[burn_in:])
    plt.plot(t[burn_in:], errs[burn_in:], label="Signal/Output Error")
    plt.vlines(tstop,0, ymax, label="Training Stop")
    plt.ylim(0,ymax)
    plt.xlabel('time', fontweight='bold', fontsize=16)
    plt.ylabel('error', fontweight='bold', fontsize=16)
    plt.title(title, fontweight='bold', fontsize=20)
    plt.legend()

def cross_signal_error(errs1, errs2, t, tstop, title, burn_in=0, mean=True, algo1="Force", algo2="SFORCE"):
    """
    errs1: list[ndarray] (mean=True) / ndarray (mean=False)
    errs2: list[ndarray] (mean=True) / ndarray (mean=False)
    t: ndarray
    tstop: float
    title: string
    burn_in: float (optional)
    mean: bool (optional)
    """
    if mean:
        errs1 = np.mean(errs1, axis=0)
        errs2 = np.mean(errs2, axis=0)
    ymax = 2*np.max([np.max(errs1[burn_in:]), np.max(errs2[burn_in:])])
    plt.plot(t[burn_in:], errs1[burn_in:], label="S/O Error (%s)" % algo1, alpha=.8)
    plt.plot(t[burn_in:], errs2[burn_in:], label="S/O Error (%s)" % algo2, alpha=.8)
    plt.vlines(tstop,0, ymax, label="Training Stop")
    plt.ylim(0,ymax)
    plt.xlabel('time', fontweight='bold', fontsize=16)
    plt.ylabel('error', fontweight='bold', fontsize=16)
    plt.title(title, fontweight='bold', fontsize=20)
    plt.legend()

def target_vs_output_plus_error(t, z, wu, target, offset=0, log=True):
    """
    t: ndarray
    z: ndarray
    wu: ndarray
    target: ndarray/func
    offset: float
    log: bool
    """
    plt.subplot(2, 1, 1)
    if isinstance(target, types.FunctionType):
        plt.plot(t, target(t), '-r', lw=2)
    else:
        plt.plot(t[offset:], target, '-r', lw=2)
    plt.plot(t, z, '-b')
    plt.legend(('target', 'output'))
    plt.ylim([-1.1, 3])
    plt.xticks([])
    plt.subplot(2, 1, 2)
    plt.plot(t, wu, '-k')
    if log:
        plt.yscale('log')
    plt.ylabel('$|\Delta w|$', fontsize=20)
    plt.xlabel('time', fontweight='bold', fontsize=16)

def visualize_3dim_state(time, tv, pca_x):
    """
    3-dim PCA state visualization for IPython widget; called via interact.
    NOTE: interact widget only works inside Juypter notebook.

    Usage:
    > rom ipywidgets import interact, fixed
    > ...
    > tmax = t[-1]
    > tmin = t[0]
    > tv = t
    > interact(visualize_3dim_state, time=(tmin, tmax, .1), pca_x=fixed(pca_x), tv=fixed(tv));

    Parameters
    ----------
    time: ndarray
    tv: float
    pca_x: ndarray
    """
    ti = np.argmax(tv >= time)
    ax = plt.figure(figsize=(10,10)).add_subplot(111, projection='3d')
    ax.scatter(pca_x[0][ti], pca_x[1][ti], pca_x[2][ti], c='red');
    plt.plot(pca_x[0][:ti], pca_x[1][:ti], pca_x[2][:ti]);
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')


def visualize_2dim_state(time, tv, pca_x):
    """
    2-dim PCA state visualization for IPython widget; called via interact.
    NOTE: interact widget only works inside Juypter notebook.

    Usage:
    > from ipywidgets import interact, fixed
    > ...
    > tmax = t[-1]
    > tmin = t[0]
    > tv = t[:500]
    > interact(visualize_2dim_state, time=(tmin, tmax, .1), pca_x=fixed(pca_x), tv=fixed(tv));

    Parametersw
    ----------
    time: ndarray
    tv: float
    pca_x: ndarray
    """
    ti = np.argmax(tv >= time)
    plt.figure(figsize=(6,6))
    plt.plot(pca_x[0][:ti], pca_x[1][:ti])
    plt.plot(pca_x[0][ti], pca_x[1][ti], 'ro')
    plt.ylabel("PCA 2")
    plt.xlabel("PCA 1")
