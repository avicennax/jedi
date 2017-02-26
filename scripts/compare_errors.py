import numpy as np
import matplotlib.pyplot as plt
from jedi import jedi
from jedi.utils import plot, seedutil, func_generator, init_tools
from jedi.utils import stats as cohen
import cPickle
import argparse
import scipy.stats as stats
import sys

parser = argparse.ArgumentParser(description='Display errors plots for tasks')
parser.add_argument('task')
parser.add_argument('task_type')
parser.add_argument('noise', nargs="?")

def run(path, title):
    ds = cPickle.load(open(path, "rb"))
    errors = np.array(ds['force'][0])**2
    derrors = np.array(ds['dforce'][0])**2
    params = ds['parameters']

    tstart, tstop, t = params['tstart'], params['tstop'], params['t']

    # Post-training period error

    ti = np.argmax(t > tstop)

    post_train_e = errors[:, ti:650]
    post_train_de = derrors[:, ti:650]

    del params['t']
    print params
    print ("FORCE --")
    print ("Error mean: %.4f" % np.mean(post_train_e ))
    print ("Error std: %.4f\n" % np.std(post_train_e ))


    print ("DFORCE --")
    print ("Error mean: %.4f" % np.mean(post_train_de))
    print ("Error std: %.4f\n" % np.std(post_train_de))

    print("t-test: %.2E" %
          stats.ttest_ind(np.concatenate(post_train_e),
                          np.concatenate(post_train_de), equal_var=False)[1])
    print("Cohen's d: %.2f" % cohen.cohen_d(np.concatenate(post_train_e),
                                           np.concatenate(post_train_de)) )


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
