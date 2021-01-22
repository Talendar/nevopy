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

"""
TODO
"""

from typing import Any, List, Optional
from nevopy.base_genome import BaseGenome, IncompatibleGenomesError
from nevopy.fixed_topology.layers import BaseLayer, IncompatibleLayersError
from nevopy.fixed_topology.config import FixedTopologyConfig


class FixedTopologyGenome(BaseGenome):
    """
    TODO
    """

    def __init__(self,
                 layers: List[BaseLayer],
                 config: Optional[FixedTopologyConfig] = None) -> None:
        super().__init__()
        # todo: automatically set hidden layers and output layer input shape
        self.layers = layers
        self.config = config if config is not None else FixedTopologyConfig()

    def process(self, X: Any) -> Any:
        prev_output = X
        for layer in self.layers:
            prev_output = layer(prev_output)
        return prev_output

    def reset(self) -> None:
        """ This method doesn't do anything.

        In this implementation, the default fixed-topology networks do not need
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
        """ TODO

        Args:
            other:

        Returns:

        Raises:
            IncompatibleGenomesError:
        """
        if len(self.layers) != len(other.layers):
            raise IncompatibleGenomesError("Attempt to mate genomes with a"
                                           "different number of layers! "
                                           f"Expected {len(self.layers)} "
                                           f"layers, got {len(other.layers)}.")

        new_layers = []

        # exchange weights mode
        if self.config.mating_mode == "exchange_weights":
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
            pass
        # invalid mating mode
        else:
            raise ValueError(f"Invalid mating mode "
                             f"(\"{self.config.mating_mode}\")!")

        return FixedTopologyGenome(layers=new_layers,
                                   config=self.config)
