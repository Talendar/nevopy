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

""" Implements genomes (subclasses of :class:`.BaseGenome`) that encode neural
networks with a fixed topology.
"""

import logging
from typing import Any, List, Optional, Tuple

import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential as KerasSequential
from tensorflow.keras.utils import plot_model as keras_plot_model

from nevopy.base_genome import BaseGenome, IncompatibleGenomesError
from nevopy.fixed_topology.layers.base_layer import BaseLayer
from nevopy.fixed_topology.layers.tf_layers import IncompatibleLayersError
from nevopy.fixed_topology.layers.tf_layers import TensorFlowLayer
from nevopy.genetic_algorithm.config import GeneticAlgorithmConfig

_logger = logging.getLogger(__name__)


class FixedTopologyGenome(BaseGenome):
    """ Genome that encodes a fixed-topology multilayer neural network.

    This genome directly encodes a multilayer neural network with fixed
    topology. The network is defined by its layers (instances of a subclass of
    :class:`.BaseLayer`), specified during the genome's creation.

    Note:
        The `config` objects of individual layers are forcefully replaced by the
        `config` object of the genome when its assigned with a new one!

    Args:
        layers (List[BaseLayer]): List with the layers of the network (instances
            of a subclass of :class:`.BaseLayer`). It's not required to set the
            input shape of each individual layer. If the input shapes are not
            set, they will be automatically set when a call to
            :meth:`.process()` is made. There is no need to pass the `config`
            object to the layers (it's done automatically when this class is
            instantiated).
        config (Optional[GeneticAlgorithmConfig]): Settings of the current
            evolutionary session. If `None`, a config object must be assigned to
            this genome latter.
        input_shape (Optional[Tuple[int, ...]]): Shape of the inputs that will
            be fed to the genome. If a value is specified, the genome's layers
            are built (they have their weights initialized). If `None`, an
            input shape will be inferred later when an input is fed to the
            genome (note, however, that the weights won't be initialized until
            it occurs).

    Attributes:
        layers (List[BaseLayer]): List with the layers of the network (instances
            of a subclass of :class:`.BaseLayer`).
    """

    def __init__(self,
                 layers: List[BaseLayer],
                 config: Optional[GeneticAlgorithmConfig] = None,
                 input_shape: Optional[Tuple[int, ...]] = None) -> None:
        super().__init__()
        self.layers = layers
        self._config = None
        self.config = config

        if input_shape is not None:
            if layers[0].input_shape is None:
                _logger.info(" Feeding test data to genome in order to build "
                             f"its layers (input shape: {input_shape})")
                self.process(np.zeros(shape=input_shape))
            elif layers[0].input_shape != input_shape:
                raise ValueError("The input shape passed as argument doesn't "
                                 "match the input shape of the genome's first "
                                 "layer!")

    @property
    def input_shape(self) -> Optional[Tuple[int, ...]]:
        """ The input shape expected by the genome's input layer. """
        return self.layers[0].input_shape

    @property
    def config(self) -> Optional[GeneticAlgorithmConfig]:
        return self._config

    @config.setter
    def config(self, c) -> None:
        self._config = c
        for layer in self.layers:
            layer.config = self._config

    def process(self, x: Any) -> Any:
        prev_output = x
        for layer in self.layers:
            prev_output = layer(prev_output)
        return prev_output

    def reset(self) -> None:
        """ This method doesn't do anything.

        In this implementation, the default fixed topology networks do not need
        to reset any of its internal states before the start of a new
        generation.
        """
        pass

    def mutate_weights(self) -> None:
        """ Randomly mutates the weights of the genome's connections. """
        for layer in self.layers:
            layer.mutate_weights()

    def random_copy(self) -> "FixedTopologyGenome":
        return FixedTopologyGenome(layers=[layer.random_copy()
                                           for layer in self.layers],
                                   config=self.config)

    def deep_copy(self) -> "FixedTopologyGenome":
        return FixedTopologyGenome(layers=[layer.deep_copy()
                                           for layer in self.layers],
                                   config=self.config)

    def mate(self, other: "FixedTopologyGenome") -> "FixedTopologyGenome":
        """ Mates two genomes to produce a new genome (offspring).

        Implements the sexual reproduction between a pair of genomes. The new
        genome inherits information from both parents.

        Currently available mating modes for individual layers:

            * :func:`.mating.exchange_weights_mating`;
            * :func:`.mating.exchange_units_mating`;
            * :func:`.mating.weights_avg_mating`.

        The mating mode of a layer is specified during its instantiation.

        Args:
            other (Any): The second genome . If it's not compatible for mating
                with the current genome (`self`), an exception will be raised.

        Returns:
            A new genome (the offspring born from the sexual reproduction
            between the current genome and the genome passed as argument).

        Raises:
            IncompatibleGenomesError: If the genome passed as argument to
                ``other`` is incompatible with the current genome (`self`).
        """
        if len(self.layers) != len(other.layers):
            raise IncompatibleGenomesError("Attempt to mate genomes with a"
                                           "different number of layers! "
                                           f"Expected {len(self.layers)} "
                                           f"layers, got {len(other.layers)}.")

        new_layers = []

        # Exchange weights mode
        if self.config.mating_mode == "weights_mating":
            for layer1, layer2 in zip(self.layers, other.layers):
                try:
                    new_layers.append(layer1.mate(layer2))
                except IncompatibleLayersError as e:
                    raise IncompatibleGenomesError(
                        "Attempt to mate a layer of a genome with an "
                        "incompatible layer of another genome!"
                    ) from e
        # Exchange layers mode
        elif self.config.mating_mode == "exchange_layers":
            parents = (self, other)
            chosen_parents = np.random.choice([0, 1],
                                              size=len(self.layers), p=[.5, .5])
            for idx, p in enumerate(chosen_parents):
                new_layers.append(parents[p].layers[idx].deep_copy())
        # Invalid mating mode
        else:
            raise ValueError(f"Invalid mating mode "
                             f"(\"{self.config.mating_mode}\")!")

        return FixedTopologyGenome(layers=new_layers,
                                   config=self.config)

    def distance(self, other: "FixedTopologyGenome") -> float:
        """ Calculates the distance between the two genomes.

        The distance is calculated based on the euclidean distance (the L2 norm
        of the difference) between correspondent weight matrices of the genomes
        layers.

        Args:
            other (FixedTopologyGenome): The other fixed-topology genome.

        Returns:
            A float representing the distance between the two genomes. The lower
            the distance, the more similar the two genomes are.
        """
        total_dist = 0.0
        for l1, l2 in zip(self.layers, other.layers):
            layers_dist = 0.0

            c = 0
            for c, (w1, w2) in enumerate(zip(l1.weights, l2.weights)):
                norm = np.linalg.norm(w1 - w2)
                layers_dist += norm / np.sqrt(w1.size)

            if c > 0:
                layers_dist /= c

            total_dist += layers_dist

        return total_dist / len(self.layers)

    def visualize(self,
                  show: bool = True,
                  to_file: str = "genome.png",
                  **kwargs) -> Image.Image:
        """ Utility method for visualizing the genome's neural network.

        This currently only works with genomes that use TensorFlow layers.

        Todo:
            Make it possible to visualize neurons and connections.

        Attributes:
            show (bool): Whether to show the generated image or not.
            to_file (str): Path in which the image file will be saved to.
            **kwargs: Optional named arguments to be passed to
                :py:func:`tensorflow.keras.utils.plot_model`.

        Returns:
            The generated ``PIL.Image.Image`` object.
        """
        # Checking compatibility:
        for layer in self.layers:
            if not isinstance(layer, TensorFlowLayer):
                raise ValueError("The genome has incompatible layers! "
                                 "Currently, it's only possible to visualize "
                                 "TensorFlow's layers.")

        # Building keras model and visualizing it:
        # noinspection PyUnresolvedReferences
        model = KerasSequential(layers=[layer.tf_layer  # type: ignore
                                        for layer in self.layers])
        model(np.zeros(shape=self.input_shape))
        keras_plot_model(model, show_shapes=True, to_file=to_file, **kwargs)

        img = Image.open(to_file)
        if show:
            img.show()

        return img
