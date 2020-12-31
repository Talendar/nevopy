"""
Utility functions.
"""

import numpy as np


def chance(p):
    """ Randomly returns True or False. The parameter specifies the chance of returning True."""
    return np.random.uniform(low=0, high=1) < p
