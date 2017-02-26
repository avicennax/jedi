import numpy as np

def cohen_d(x1, x2):
    """
    Computes Cohen's d; an effect size statistic:
    Overview: https://en.wikipedia.org/wiki/Effect_size#Cohen.27s_d
    """
    n1 = len(x1)
    n2 = len(x2)
    return (np.mean(x1) - np.mean(x2)) / np.sqrt(((n1-1)*np.var(x1) + (n2-1)*np.var(x2)) / (n1 + n2-2))
