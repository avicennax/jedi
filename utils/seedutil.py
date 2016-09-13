# seedutil.py
# Various environment helper functions for random
# seeds, including saving and loading seeds.

from __future__ import division
import os.path
import datetime
import csv


import numpy as np

def save_seeds(seeds, description='', filename=None, dir='../data', seed_key=None):
    t = datetime.datetime.now()
    if filename is None:
        ts = ['seeds', t.year, t.month, t.day, t.hour, t.second]
        filename = '_'.join([str(t) for t in ts])
    path = os.path.join(dir, filename)
    np.save(path, seeds)
    _insert_csv(seeds, description, filename, seed_key, dir)

def _insert_csv(seeds, description, filename, seed_key, dir):
    """
    Documentations for different seeds.
    """
    rows = []
    path = os.path.join(dir,'seeds.csv')

    if os.path.isfile(path):
         with open(path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                rows.append(row)

    if seed_key is None:
        if len(rows) != 0:
            seed_key = int(rows[-1][0])+1 # Using last seed key + 1
        else:
            seed_key = 0

    keys = [row[0] for row in rows]
    while seed_key in keys:
        seed_key += 1

    print "Seed Key: %d" % seed_key

    new_row = [seed_key, filename, str(seeds), description]

    with open(path, 'w') as csvfile:
        fieldnames = ['key', 'filename', 'seeds', 'description']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        for row in rows:
            writer.writerow({fn:f for fn, f in zip(fieldnames, row)})
        writer.writerow({fn:f for fn, f in zip(fieldnames, new_row)})


def load_seeds(filename, dir='../data'):
    path = os.path.join(dir, filename)
    return np.load(path)
