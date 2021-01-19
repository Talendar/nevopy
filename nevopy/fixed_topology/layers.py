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

""" Implements neural network layers.

This module implements a variety of neural network layers to be used by genomes
in the context of neuroevolution.
"""

from abc import ABC, abstractmethod
from typing import Any
import tensorflow as tf


class BaseLayer(ABC):
    """ Abstract base class that defines a neural network layer.

    This abstract base class defines the general structure and behaviour of a
    neural network layer in the context of neuroevolutionary algorithms.
    """

    def __init__(self):
        pass

    @property
    @abstractmethod
    def weights(self) -> Any:
        """ The layer's weights.

        Usually contained within a `NumPy ndarray` or a `TensorFlow tensor`.
        """

    @abstractmethod
    def process(self, X: Any) -> Any:
        """ Feeds the given input(s) to the layer.

        This is where the layer's logic lives.

        Args:
            X (Any): The input(s) to be fed to the layer. Usually a
                `NumPy ndarray` or a `TensorFlow tensor`.

        Returns:
            The output of the layer. Usually a `NumPy ndarray` or a
            `TensorFlow tensor`.
        """

    def __call__(self, X: Any) -> Any:
        """ Wraps a call to :meth:`.process`. """
        return self.process(X)

    @abstractmethod
    def deep_copy(self) -> "BaseLayer":
        """ Makes an exact/deep copy of the layer.

        Returns:
            An exact/deep copy of the layer.
        """

    @abstractmethod
    def mutate_weights(self) -> None:
        """ Randomly mutates the weights of the genome's connections. """

    @abstractmethod
    def mate(self, other: Any) -> "BaseLayer":
        """ Mates two layers to produce a new layer (offspring).

        Implements the sexual reproduction between a pair of layers. The new
        layer inherits information from both parents (not necessarily in an
        equal proportion)

        Args:
            other (Any): The second layer . If it's not compatible for mating
                with the current layer (`self`), an exception will be raised.

        Returns:
            A new layer (the offspring born from the sexual reproduction between
            the current layer and the layer passed as argument.

        Raises:
            IncompatibleLayersError: If the layer passed as argument to
                ``other`` is incompatible with the current layer (`self`).
        """


class TensorFlowLayer(BaseLayer, ABC):
    """ Abstract base class for layers that wrap a `TensorFlow` layer. """

    @property
    @abstractmethod
    def tf_layer(self) -> tf.keras.layers.Layer:
        """
        The `tf.keras.layers.Layer
        <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer>`_
        used internally.
        """


class TFConv2DLayer(TensorFlowLayer):
    """ Wraps a `TensorFlow` 2D convolution layer.

    This is a wrapper for `tf.keras.layers.Conv2D
    <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D>`_.
    """

    # TODO


class IncompatibleLayersError(Exception):
    """
    Indicates that an attempt has been made to mate (sexual reproduction) two
    incompatible layers.
    """