""" Pre-implemented activation functions.

"""

import numpy as np


def linear(x):
    """ Linear activation function (simply returns the input, unchanged)."""
    return x


def sigmoid(x, clip_value=64):
    """ Numeric stable implementation of the sigmoid function.

    Estimated lower-bound precision with a clip value of 64: 10^(-28)
    """
    x = np.clip(x, -clip_value, clip_value)
    return 1 / (1 + np.exp(-x))


def steepened_sigmoid(x, step=4.9):
    """ Steepened version of the sigmoid function.

    The original NEAT paper used a steepened version of the sigmoid function with a step value of 4.9.

    "We used a modified sigmoidal transfer function, ϕ(x) = 1 / (1 + exp(−4.9x)), at all nodes. The steepened sigmoid
    allows more fine tuning at extreme activations. It is optimized to be close to linear during its steepest ascent
    between activations −0.5 and 0.5."
    - Stanley, K. O. & Miikkulainen, R. (2002)
    """
    return sigmoid(x * step)
