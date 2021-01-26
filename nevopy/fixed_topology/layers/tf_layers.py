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

""" Implements subclasses of :class:`.BaseLayer` that wrap TensorFlow layers.
"""

import numpy as np
from typing import Any, Tuple, Dict, Optional, Union, Type, Callable, List

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "1"
import tensorflow as tf

from nevopy.base_genome import InvalidInputError
from nevopy.fixed_topology.config import FixedTopologyConfig
from nevopy.fixed_topology.layers import mating
from nevopy.fixed_topology.layers.base_layer import BaseLayer
from nevopy.fixed_topology.layers.base_layer import IncompatibleLayersError


class TensorFlowLayer(BaseLayer):
    """ Wraps a `TensorFlow` layer.

    This class wraps a `TensorFlow` layer, making it compatible with `NEvoPY's`
    neuroevolution algorithms. It handles the mutation and reproduction of the
    `TensorFlow` layer.

    In most cases, there is no need to create subclasses of this class. Doing
    that to frequently used types of layers, however, may be desirable, since it
    makes using those types of layers easier (see :class:`.TFConv2DLayer` and
    :class:`.TFDenseLayer` as examples).

    When inheriting this class, you'll usually do something like this:

        .. code-block:: python

            class MyTFLayer(TensorFlowLayer):
                def __init__(self,
                             arg1, arg2,
                             activation="relu",
                             mating_func = mating.exchange_units_mating,
                             config = None,
                             input_shape = None,
                             mutable = True,
                             **tf_kwargs: Dict[str, Any]) -> None:
                    super().__init__(
                        layer_type=tf.keras.layers.SomeKerasLayer,
                        **{k: v for k, v in locals().items()
                           if k != "self" and k != "tf_kwargs" and k != "__class__"},
                        **tf_kwargs,
                    )

    Args:
        layer_type (Union[str, Type[tf.keras.layers.Layer]]): A reference to the
            `TensorFlow's` class that represents the layer
            (:py:class:`tf.keras.layers.Dense`, for example). If it's a string,
            the appropriate type will be inferred (note that it must be listed
            in :attr:`.TensorFlowLayer.KERAS_LAYERS`).
        mating_func (Optional[Callable[[BaseLayer, BaseLayer], BaseLayer]]):
            Function that mates (sexual reproduction) two layers. It should
            receive two layers as input and return a new layer (the offspring).
            You can use one of the pre-built mating functions (see
            :mod:`.fixed_topology.layers.mating`) or implement your own. If the
            layer is immutable, this parameter should receive `None` as
            argument.
        config (Optional[FixedTopologyConfig]): Settings being used in the
            current evolutionary session. If `None`, a config object must be
            assigned to the layer later on, before calling the methods that
            require it.
        input_shape (Optional[Tuple[int, ...]]): Shape of the data that will be
            processed by the layer. If `None`, an input shape for the layer must
            be manually specified later or be inferred from an input sample.
        mutable (Optional[bool]): Whether or not the layer can have its weights
            changed (mutation).
        **tf_kwargs: Named arguments to be passed to the constructor of the
            `TensorFlow` layer.
    """

    KERAS_LAYERS = {
        "conv2D": tf.keras.layers.Conv2D,
        "dense": tf.keras.layers.Dense,
        "flatten": tf.keras.layers.Flatten,
        "simple_rnn": tf.keras.layers.SimpleRNN,
        "rnn": tf.keras.layers.RNN,
        "lstm": tf.keras.layers.LSTM,
    }

    def __init__(self,
                 layer_type: Union[str, Type[tf.keras.layers.Layer]],
                 mating_func: Optional[
                     Callable[[BaseLayer, BaseLayer], BaseLayer]
                 ] = mating.exchange_units_mating,
                 config: Optional[FixedTopologyConfig] = None,
                 input_shape: Optional[Tuple[int, ...]] = None,
                 mutable: Optional[bool] = True,
                 **tf_kwargs) -> None:
        super().__init__(config, input_shape, mutable)
        self._layer_type = (layer_type if not isinstance(layer_type, str)
                            else TensorFlowLayer.KERAS_LAYERS[layer_type])
        self._tf_layer_kwargs = tf_kwargs
        self.mating_func = mating_func

        self._tf_layer = layer_type(**self._tf_layer_kwargs)  # type: ignore
        if input_shape is not None:
            self.build(input_shape)

    @property
    def tf_layer(self) -> tf.keras.layers.Layer:
        """
        The `tf.keras.layers.Layer
        <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer>`_
        used internally.
        """
        return self._tf_layer

    @property
    def weights(self) -> List[np.ndarray]:
        """ The current weight matrices of the layer.

        Wrapper for :meth:`tf.keras.layers.Layer.get_weights`.

        The weights of a layer represent the state of the layer. This property
        returns the weight values associated with this layer as a list of Numpy
        arrays. In most cases, it's a list containing the weights of the layer's
        connections and the bias values (one for each neuron, generally).
        """
        return [w.numpy() for w in self.tf_layer.weights]

    @weights.setter
    def weights(self, new_weights: List[np.ndarray]) -> None:
        """ Wrapper for :meth:`tf.keras.layers.Layer.set_weights()`. """
        self.tf_layer.set_weights(new_weights)

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """ Wrapper for :meth:`tf.keras.layers.Layer.build()`. """
        self.tf_layer.build(input_shape=input_shape)
        self._input_shape = input_shape

    def process(self, X: Union[np.ndarray, tf.Tensor]) -> tf.Tensor:
        try:
            result = self.tf_layer(X)
            if self._input_shape is None:
                self._input_shape = X.shape
            return result
        except ValueError as e:
            raise InvalidInputError("The given input's shape doesn't match the "
                                    "shape expected by the layer! "
                                    f"TensorFlow's error message: {str(e)}")

    def _new_instance(self):
        """ Returns a new instance of the layer.

        The new instance doesn't inherit the current layer's weights - a new set
        of weights is initialized.
        """
        return TensorFlowLayer(layer_type=self._layer_type,
                               mating_func=self.mating_func,
                               config=self.config,
                               input_shape=self._input_shape,
                               mutable=self.mutable,
                               **self._tf_layer_kwargs)

    def random_copy(self) -> "TensorFlowLayer":
        if not self.mutable:
            return self.deep_copy()
        return self._new_instance()

    def deep_copy(self) -> "TensorFlowLayer":
        new_layer = self._new_instance()
        new_layer.weights = self.weights
        return new_layer

    def mutate_weights(self, _test_info=None) -> None:
        """ Randomly mutates the weights of the layer's connections.

        Each weight will be perturbed by an amount defined in the settings of
        the current evolutionary session. Each weight also has a chance of being
        reset (a new random value is assigned to it).

        If the layer is immutable, nothing happens (the layer's weights remain
        unchanged).
        """
        if not self.mutable:
            return

        assert self.config is not None
        if self.input_shape is None:
            raise RuntimeError("Attempt to mutate the weights of a layer that "
                               "didn't have its weight and bias matrices "
                               "initialized!")

        new_weights = []
        for i, w in enumerate(self.weights):
            # weight perturbation
            w_perturbation = tf.random.uniform(
                shape=w.shape,
                minval=1 - self.config.weight_perturbation_pc,
                maxval=1 + self.config.weight_perturbation_pc,
            )
            w = tf.math.multiply(w, w_perturbation).numpy()

            # weight reset
            num_w_reset = np.random.binomial(w.size,
                                             self.config.weight_reset_chance)
            if num_w_reset > 0:
                w_reset_idx = np.random.randint(0, w.size, size=num_w_reset)
                w.flat[w_reset_idx] = np.random.uniform(
                    low=self.config.new_weight_interval[0],
                    high=self.config.new_weight_interval[1],
                    size=num_w_reset,
                )

            # saving weight matrix
            new_weights.append(w)

            # test info
            if _test_info is not None:
                _test_info[f"w{i}_perturbation"] = w_perturbation
                # noinspection PyUnboundLocalVariable
                _test_info[f"w{i}_reset_idx"] = (w_reset_idx if num_w_reset > 0
                                                 else [])

        # setting new weights and biases
        self.weights = new_weights

    def mate(self, other: "TensorFlowLayer") -> "TensorFlowLayer":
        if self.mutable != other.mutable:
            raise IncompatibleLayersError("Attempt to mate an immutable "
                                          "layer with a mutable layer!")

        if other == self or not self.mutable:
            return self.deep_copy()

        return self.mating_func(self, other)  # type: ignore


class TFConv2DLayer(TensorFlowLayer):
    """ Wraps a `TensorFlow` 2D convolution layer.

    This is a simple wrapper for `tf.keras.layers.Conv2D
    <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D>`_.
    """

    def __init__(self,
                 filters: int,
                 kernel_size: Tuple[int, int],
                 strides: Tuple[int, int] = (1, 1),
                 padding: str = "valid",
                 activation="relu",
                 mating_func: Optional[
                     Callable[[BaseLayer, BaseLayer], BaseLayer]
                 ] = mating.exchange_units_mating,
                 config: Optional[FixedTopologyConfig] = None,
                 input_shape: Optional[Tuple[int, ...]] = None,
                 mutable: Optional[bool] = True,
                 **tf_kwargs: Dict[str, Any]) -> None:
        super().__init__(
            layer_type=tf.keras.layers.Conv2D,
            **{k: v for k, v in locals().items()
               if k != "self" and k != "tf_kwargs" and k != "__class__"},
            **tf_kwargs,
        )


class TFDenseLayer(TensorFlowLayer):
    """ Wraps a `TensorFlow` dense layer.

    This is a simple wrapper for `tf.keras.layers.Dense
    <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense>`_.
    """

    def __init__(self,
                 units: int,
                 activation=None,
                 mating_func: Optional[
                     Callable[[BaseLayer, BaseLayer], BaseLayer]
                 ] = mating.exchange_units_mating,
                 config: Optional[FixedTopologyConfig] = None,
                 input_shape: Optional[Tuple[int, ...]] = None,
                 mutable: Optional[bool] = True,
                 **tf_kwargs: Dict[str, Any]) -> None:
        super().__init__(
            layer_type=tf.keras.layers.Dense,
            **{k: v for k, v in locals().items()
               if k != "self" and k != "tf_kwargs" and k != "__class__"},
            **tf_kwargs,
        )


class TFFlattenLayer(TensorFlowLayer):
    """ Wraps a `TensorFlow` flatten layer.

    This is a simple wrapper for `tf.keras.layers.Flatten
    <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten>`_.
    """

    def __init__(self,
                 mating_func: Optional[
                     Callable[[BaseLayer, BaseLayer], BaseLayer]
                 ] = None,
                 config: Optional[FixedTopologyConfig] = None,
                 input_shape: Optional[Tuple[int, ...]] = None,
                 mutable: Optional[bool] = False,
                 **tf_kwargs: Dict[str, Any]) -> None:
        super().__init__(
            layer_type=tf.keras.layers.Flatten,
            **{k: v for k, v in locals().items()
               if k != "self" and k != "tf_kwargs" and k != "__class__"},
            **tf_kwargs,
        )
