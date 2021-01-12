# MIT License
#
# Copyright (c) 2020 Gabriel Nogueira (Talendar)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

""" This module implements some activation functions.

Todo:
    Make all activation functions compatible with numpy arrays.
"""

import numpy as np


def linear(x: float) -> float:
    """ Linear activation function (simply returns the input, unchanged). """
    return x


def sigmoid(x: float,
            clip_value: int = 64) -> float:
    """ Numeric stable implementation of the sigmoid function.

    Estimated lower-bound precision with a clip value of 64: 10^(-28).
    """
    x = np.clip(x, -clip_value, clip_value)
    return 1 / (1 + np.exp(-x))


def steepened_sigmoid(x: float,
                      step: float = 4.9) -> float:
    """ Steepened version of the sigmoid function.

    The original NEAT paper used a steepened version of the sigmoid function
    with a step value of 4.9.

    "We used a modified sigmoidal transfer function,
    ϕ(x) = 1 / (1 + exp(−4.9x)), at all nodes. The steepened sigmoid allows more
    fine tuning at extreme activations. It is optimized to be close to linear
    during its steepest ascent between activations −0.5 and 0.5."
    - :cite:`stanley:ec02`
    """
    return sigmoid(x * step)
