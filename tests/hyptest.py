import hypertools as hyp
import numpy as np

x = np.load('/Users/simonhaxby/Code/Python/jedi/data/random/act.npy')
hyp.plot([x], animate=True)