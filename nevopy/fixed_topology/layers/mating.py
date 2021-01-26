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

""" Implements some mating (sexual reproduction) functions that can be used to
generate a new neural network layer from two parent layers.

Todo:
    Make the mating functions compatible with :class:`.BaseLayer`, not just
    :class:`TensorFlowLayer`.
"""

from typing import List
import numpy as np
import random

from nevopy.fixed_topology.layers.base_layer import BaseLayer
from nevopy.fixed_topology.layers.base_layer import IncompatibleLayersError


def check_weights_compatibility(weight_list1: List[np.ndarray],
                                weight_list2: List[np.ndarray]):
    """ TODO

    Args:
        weight_list1:
        weight_list2:

    Raises:
        IncompatibleLayersError:
    """
    if len(weight_list1) != len(weight_list2):
        raise IncompatibleLayersError("The layers have weight lists of "
                                      f"different lengths! "
                                      f"Layer 1: {len(weight_list1)}. "
                                      f"Layer 2: {len(weight_list2)}.")

    for i, (w1, w2) in enumerate(zip(weight_list1, weight_list2)):
        if w1.shape != w2.shape:
            raise IncompatibleLayersError(
                f"Incompatible shape between the weight matrices at index {i} "
                f"of the layers' weight lists! Layer 1 shape: {w1.shape}. "
                f"Layer 2 shape: {w2.shape}"
            )


def exchange_weights_mating(layer1: BaseLayer,
                            layer2: BaseLayer) -> BaseLayer:
    """
    TODO
    """
    # use tf.reshape(w, [-1]) to flatten and tf.reshape to bring back to the
    # original shape after exchanging individual weights



def exchange_units_mating(layer1: BaseLayer,
                          layer2: BaseLayer) -> BaseLayer:
    """ Mates (sexual reproduction) two neural layers by exchanging units.

    The term "unit" means different things depending on the type of the layers.
    For a Conv2D layer, for instance, an unit is a filter (kernel). For a Dense
    layer, on the other hand, an unit is a neuron (including the weights of its
    connections). Bias terms are also considered "units".

    Generally, we can define an unit as being whatever you get when indexing a
    weight matrix by its last shape. Given a layer `L`, an unit of its weight
    matrix at index `w` is given by `L.weights[w][..., i]`, where `i` is the
    index of the unit (`i` is in the interval `[0, w.shape[-1][`).

    Args:
        layer1 (BaseLayer): An instance of a subclass of :class:`.BaseLayer`.
        layer2 (BaseLayer): An instance of a subclass of :class:`.BaseLayer`.

    Returns:
        A new layer that inherits information from both parents.

    Raises:
        IncompatibleLayersError: If the weight matrices of the two given layers
            are not of the same shape (i.e., the layers are not compatible for
            mating).
    """
    # retrieving weights and biases as numpy arrays
    weights1 = layer1.weights
    weights2 = layer2.weights

    # checking compatibility
    check_weights_compatibility(weights1, weights2)

    # selecting units for the new layer
    new_weights = []
    for w1, w2 in zip(weights1, weights2):
        new_units = []
        for i in range(w1.shape[-1]):
            w = random.choice((w1, w2))
            new_units.append(w[..., i])
        new_weights.append(np.stack(new_units, axis=-1))

    # building new layer and changing its weights
    new_layer = layer1.random_copy()
    new_layer.weights = new_weights

    return new_layer


def sum_and_divide_mating(layer1: BaseLayer,
                          layer2: BaseLayer) -> BaseLayer:
    """
    TODO

    Args:
        layer1:
        layer2:

    Returns:

    """
    raise NotImplementedError()
