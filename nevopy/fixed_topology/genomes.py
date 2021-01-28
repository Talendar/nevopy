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

from typing import Any, List, Optional, Tuple
from nevopy.base_genome import BaseGenome, IncompatibleGenomesError
from nevopy.fixed_topology.layers import BaseLayer, IncompatibleLayersError
from nevopy.fixed_topology.config import FixedTopologyConfig
import numpy as np


class FixedTopologyGenome(BaseGenome):
    """ Genome that encodes a fixed topology multilayer neural network.

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
        config (Optional[FixedTopologyConfig]): Settings of the current
            evolutionary session. If `None`, a config object must be assigned to
            this genome latter.

    Attributes:
        layers (List[BaseLayer]): List with the layers of the network (instances
            of a subclass of :class:`.BaseLayer`).
    """

    def __init__(self,
                 layers: List[BaseLayer],
                 config: Optional[FixedTopologyConfig] = None) -> None:
        super().__init__()
        self.layers = layers
        self._config = None
        self.config = config

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """ The input shape expected by the genome's input layer. """
        return self.layers[0].input_shape

    @property
    def config(self) -> Optional[FixedTopologyConfig]:
        """ Settings of the current evolutionary session.

        If `None`, a config object hasn't been assigned to this genome yet.
        """
        return self._config

    @config.setter
    def config(self, c) -> None:
        """ Sets the config object of the genome and all of its layers. """
        self._config = c
        for layer in self.layers:
            layer.config = self._config

    def process(self, X: Any) -> Any:
        prev_output = X
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

        Currently, there are two mating methods available. In the
        `exchange weights` mode, each of the new genome's layers has its weights
        inherited from both parents (the details on how it's done depend on the
        mating function used by the layer). In the `exchange layers` mode, each
        of the new genome's layers is an exact copy of a layer from one of the
        parent genomes. The mating mode to be used is determined by
        :class:`.FixedTopologyConfig`.

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

        # exchange weights mode
        if self.config.mating_mode == "exchange_weights_mating":
            for layer1, layer2 in zip(self.layers, other.layers):
                try:
                    new_layers.append(layer1.mate(layer2))
                except IncompatibleLayersError:
                    raise IncompatibleGenomesError(
                        "Attempt to mate a layer of a genome with an "
                        "incompatible layer of another genome!"
                    )
        # exchange layers mode
        elif self.config.mating_mode == "exchange_layers":
            parents = (self, other)
            chosen_parents = np.random.choice([0, 1],
                                              size=len(self.layers), p=[.5, .5])
            for idx, p in enumerate(chosen_parents):
                new_layers.append(parents[p].layers[idx].deep_copy())
        # invalid mating mode
        else:
            raise ValueError(f"Invalid mating mode "
                             f"(\"{self.config.mating_mode}\")!")

        return FixedTopologyGenome(layers=new_layers,
                                   config=self.config)
