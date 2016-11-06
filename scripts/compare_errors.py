import numpy as np
import matplotlib.pyplot as plt
from jedi import jedi
from jedi.utils import plot, seedutil, func_generator, init_tools
import cPickle
import argparse
import sys

parser = argparse.ArgumentParser(description='Display errors plots for tasks')
parser.add_argument('task')
parser.add_argument('task_type')
parser.add_argument('noise', nargs="?")

def run(path, title):
    ds = cPickle.load(open(path, "rb"))
    errors = ds['force'][0]
    derrors = ds['dforce'][0]
    params = ds['parameters']

    tstart, tstop, t = params['tstart'], params['tstop'], params['t']

    del params['t']
    print params

    plt.figure(figsize=(12,3))
    plot.cross_signal_error(errors, derrors, t, tstart, tstop,
                        title="FORCE vs DFORCE (%s)" % title, burn_in=10)
    plt.show()

def construct_path(a):

    common_path = "../data/stability"
    if a.task_type == 'base':
        return "/".join([common_path, a.task, a.task_type, "base.p"])
    else:
        if a.noise is None:
            raise ValueError("No noise arg passed")
        if a.task_type == 'both':
            return "/".join([common_path, a.task, a.task_type+"_noise", "noise_exin_("+a.noise+").p" ])
        else:
            return "/".join([common_path, a.task, a.task_type+"_noise", "noise_"+a.noise+".p"])

if __name__ == "__main__":

    a = parser.parse_args()
    path = construct_path(a)
    run(path, a.task)
