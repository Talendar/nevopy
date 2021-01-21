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
from typing import Any, Tuple, Dict

from nevopy.base_genome import InvalidInputError
from nevopy.fixed_topology.config import FixedTopologyConfig
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "1"
import tensorflow as tf


class BaseLayer(ABC):
    """ Abstract base class that defines a neural network layer.

    This abstract base class defines the general structure and behaviour of a
    neural network layer in the context of neuroevolutionary algorithms.

    Args:
        config (FixedTopologyConfig): Settings being used in the current
            evolutionary session.
        input_shape (Tuple[int, ...]): Shape of the data that will be processed
            by the layer.

    Attributes:
        config (FixedTopologyConfig): Settings being used in the current
            evolutionary session.
    """

    def __init__(self,
                 config: FixedTopologyConfig,
                 input_shape: Tuple[int, ...]):
        self.config = config
        self._input_shape = input_shape

    @property
    @abstractmethod
    def weights(self) -> Tuple[Any, Any]:
        """ A tuple with the layer's weights and biases.

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

        Raises:
            InvalidInputError: If the shape of ``X`` doesn't match the input
                shape expected by the layer.
        """

    def __call__(self, X: Any) -> Any:
        """ Wraps a call to :meth:`.process`. """
        return self.process(X)

    @abstractmethod
    def shallow_copy(self) -> "BaseLayer":
        """ Makes a shallow / simplified copy of the layer.

        Returns:
            A new layer with the same topology of the current layer, but with
            newly initialized weights and biases.
        """

    @abstractmethod
    def deep_copy(self) -> "BaseLayer":
        """ Makes an exact/deep copy of the layer.

        Returns:
            An exact/deep copy of the layer, including its weights and biases.
        """

    @abstractmethod
    def mutate_weights(self) -> None:
        """ Randomly mutates the weights of the layer's connections. """

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
    """ Abstract base class for layers that wrap a `TensorFlow` layer.

    When subclassing this class, be sure to call ``super().__ init __()``
    passing, as named arguments, the same arguments received by the subclass's
    `` __init__()``. These arguments will be stored in the instance variable
    ``self._num_layer_kwargs``;

    This is necessary because this class implements :meth:`.deep_copy()`. This
    method is implemented in the base class because it performs, in general, the
    same actions regardless of the internal details of each subclass.

    You'll usually do something like this:

        .. code-block:: python

            class MyTFLayer(TensorFlowLayer):
                    def __init__(self, config, input_shape,
                                 some_arg1, some_arg2, **kwargs):
                        super().__init__(
                            **{k: v for k, v in locals().items()
                               if k != "self" and k != "kwargs" and k != "__class__"},
                            **kwargs,
                        )
                        self._tf_layer = tf.keras.layers.AwesomeLayer(some_arg1,
                                                                      some_arg2,
                                                                      **kwargs)
                        self._tf_layer.build(input_shape=self._input_shape)

    Args:
        config (FixedTopologyConfig): Settings being used in the current
            evolutionary session.
        input_shape (Tuple[int, ...]): Shape of the data that will be processed
            by the layer.
        **kwargs: Named arguments to be passed to the constructor of a subclass
            of this base class when making a copy of the subclass.
    """

    def __init__(self,
                 config: FixedTopologyConfig,
                 input_shape: Tuple[int, ...],
                 **kwargs):
        super().__init__(config, input_shape)
        self._new_layer_kwargs = kwargs

    @property
    @abstractmethod
    def tf_layer(self) -> tf.keras.layers.Layer:
        """
        The `tf.keras.layers.Layer
        <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer>`_
        used internally.
        """

    @property
    def weights(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """ The current weights and biases of the layer.

        Wrapper for :py:meth:`tf.keras.layers.Layer.weights`.

        The weights of a layer represent the state of the layer. This property
        returns the weight values associated with this layer as a list of Numpy
        arrays. In most cases, it's a list containing the weights of the layer's
        connections and the bias values (one for each neuron, generally).
        """
        return self.tf_layer.weights

    @weights.setter
    def weights(self, new_weights: Tuple[Any, Any]) -> None:
        """ Wrapper for :py:`tf.keras.layers.Layer.set_weights()`. """
        self.tf_layer.set_weights(new_weights)

    def process(self, X: Any) -> tf.Tensor:
        try:
            return self.tf_layer(X)
        except ValueError as e:
            raise InvalidInputError("The given input's shape doesn't match the "
                                    "shape expected by the layer! "
                                    f"TensorFlow's error message: {str(e)}")

    def shallow_copy(self) -> "TensorFlowLayer":
        return self.__class__(config=self.config,
                              input_shape=self._input_shape,
                              **self._new_layer_kwargs)

    def deep_copy(self) -> "TensorFlowLayer":
        new_layer = self.shallow_copy()
        new_layer.weights = self.weights
        return new_layer

    def mutate_weights(self, _test_info=None) -> None:
        """ Randomly mutates the weights of the layer's connections.

        Each weight will be perturbed by an amount defined in the settings of
        the current evolutionary session. Each weight also has a chance of being
        reset (a new random value is assigned to it).
        """
        weights, bias = self.weights

        # weight perturbation
        w_perturbation = tf.random.uniform(
            shape=weights.shape,
            minval=1 - self.config.weight_perturbation_pc,
            maxval=1 + self.config.weight_perturbation_pc,
        )
        weights = tf.math.multiply(weights, w_perturbation).numpy()

        # weight reset
        num_w_reset = np.random.binomial(weights.size,
                                         self.config.weight_reset_chance)
        if num_w_reset > 0:
            w_reset_idx = np.random.randint(0, weights.size, size=num_w_reset)
            weights.flat[w_reset_idx] = np.random.uniform(
                low=self.config.new_weight_interval[0],
                high=self.config.new_weight_interval[1],
                size=num_w_reset,
            )

        # bias perturbation
        b_perturbation = tf.random.uniform(
            shape=bias.shape,
            minval=1 - self.config.bias_perturbation_pc,
            maxval=1 + self.config.bias_perturbation_pc,
        )
        bias = tf.math.multiply(bias, b_perturbation).numpy()

        # bias reset
        num_b_reset = np.random.binomial(bias.size,
                                         self.config.bias_reset_chance)
        if num_b_reset > 0:
            b_reset_idx = np.random.randint(0, bias.size,
                                            size=num_b_reset)
            bias.flat[b_reset_idx] = np.random.uniform(
                low=self.config.new_bias_interval[0],
                high=self.config.new_bias_interval[1],
                size=num_b_reset,
            )

        # setting new weights and biases
        self.weights = (weights, bias)

        # test info
        if _test_info is not None:
            _test_info["w_perturbation"] = w_perturbation
            _test_info["b_perturbation"] = b_perturbation
            # noinspection PyUnboundLocalVariable
            _test_info["w_reset_idx"] = w_reset_idx if num_w_reset > 0 else []
            # noinspection PyUnboundLocalVariable
            _test_info["b_reset_idx"] = b_reset_idx if num_b_reset > 0 else []


class TFConv2DLayer(TensorFlowLayer):
    """ Wraps a `TensorFlow` 2D convolution layer.

    This is a wrapper for `tf.keras.layers.Conv2D
    <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D>`_.

    Args:
        config (FixedTopologyConfig): Settings being used in the current
            evolutionary session.
        input_shape (Tuple[int, ...]): Shape of the data that will be processed
            by the layer.
        **kwargs: Named arguments to be passed to the constructor of the
            TensorFlow layer.
    """

    def __init__(self,
                 filters: int,
                 kernel_size: Tuple[int, int],
                 config: FixedTopologyConfig,
                 input_shape: Tuple[int, ...],
                 strides: Tuple[int, int] = (1, 1),
                 padding: str = "valid",
                 activation="relu",
                 **kwargs: Dict[str, Any]) -> None:
        super().__init__(
            **{k: v for k, v in locals().items()
               if k != "self" and k != "kwargs" and k != "__class__"},
            **kwargs,
        )
        self._tf_layer = tf.keras.layers.Conv2D(filters=filters,
                                                kernel_size=kernel_size,
                                                input_shape=self._input_shape,
                                                strides=strides,
                                                padding=padding,
                                                activation=activation,
                                                **kwargs)
        self._tf_layer.build(input_shape=self._input_shape)

    @property
    def tf_layer(self) -> tf.keras.layers.Conv2D:
        return self._tf_layer

    def mate(self, other: "TFConv2DLayer") -> "TFConv2DLayer":
        """ Mates (sexual reproduction) two TF's 2D-convolutional layers.

        Each filter (kernel) and bias of the new layer is inherited, randomly
        and with an equal chance, from one of the parents.

        Args:
            other (TFConv2DLayer): The "sexual partner" of the current layer.

        Returns:
            A new 2D-convolutional layer that inherits information from both
            parents.

        Raises:
            IncompatibleLayersError: If the weight and bias matrices of the two
                given layers are not of the same shape (i.e., the layers are not
                compatible for mating).
        """
        # retrieving weights and biases as numpy arrays
        f_array1, f_array2 = self.weights[0].numpy(), other.weights[0].numpy()
        b_array1, b_array2 = self.weights[1].numpy(), other.weights[1].numpy()

        # checking compatibility
        if (f_array1.shape != f_array2.shape
                or b_array1.shape != b_array2.shape):
            raise IncompatibleLayersError(
                "The given layer is not a valid mate (sexual partner)!\n"
                f"Weights: expected shape {f_array1.shape}, got shape "
                f"{f_array2.shape}.\nBiases: expected shape {b_array1.shape}, "
                f"got shape {b_array2.shape}."
            )

        # isolating filters
        filters1 = [f_array1[:, :, :, i] for i in range(f_array1.shape[-1])]
        filters2 = [f_array2[:, :, :, i] for i in range(f_array2.shape[-1])]

        # selecting filters and biases for the new layer
        f_parents = (filters1, filters2)
        b_parents = (b_array1, b_array2)

        new_filters, new_biases = [], []
        chosen_parents = np.random.choice([0, 1],
                                          size=len(filters1), p=[.5, .5])
        for idx, p in enumerate(chosen_parents):
            f, b = f_parents[p][idx], b_parents[p][idx]
            new_filters.append(f)
            new_biases.append(b)

        # building new layer and changing its weights
        new_layer = self.shallow_copy()
        new_layer.weights = (np.stack(new_filters, axis=-1),
                             np.array(new_biases))
        return new_layer  # type: ignore


class IncompatibleLayersError(Exception):
    """
    Indicates that an attempt has been made to mate (sexual reproduction) two
    incompatible layers.
    """