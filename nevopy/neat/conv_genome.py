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

from typing import cast, Sequence
import numpy as np
from nevopy import neat
from nevopy.neat.genome import Genome


class StaticConvGenome(neat.genome.Genome):
    """ Fixed topology convolutional genome.

    The main difference between this genome and :class:`.neat.genome.Genome` is
    that before feeding the input to the regular NEAT graph neural network, it
    pre-processes the input with a convolutional neural network (CNN). The CNN
    can be evolved by having its weights mutated, but its topology is fixed.

    TODO
    """

    def shallow_copy(self, new_genome_id: int) -> "StaticConvGenome":
        """ TODO

        """
        new_genome = StaticConvGenome(genome_id=new_genome_id,
                                      num_inputs=len(self._input_nodes),
                                      num_outputs=len(self._output_nodes),
                                      config=self.config,
                                      initial_connections=False)
        return new_genome

    def deep_copy(self, new_genome_id: int) -> "StaticConvGenome":
        """ TODO

        """
        # `Genome.deep_copy()` uses the `shallow_copy` method (overridden above)
        # to create the new genome, so we can be sure that its return type is,
        # in our case, an instance of `StaticConvGenome`.
        new_genome = cast(StaticConvGenome, super().deep_copy(new_genome_id))
        return new_genome

    def distance(self, other: Genome) -> float:
        """ TODO: should the CNN be considered when calculating the distance
             between two convolutional genomes?
        """
        return super().distance(other)

    @classmethod
    def random_genome(cls,
                      num_inputs: int,
                      num_outputs: int,
                      config: neat.config.Config,
                      id_handler: neat.id_handler.IdHandler,
                      max_hidden_nodes: int,
                      max_hidden_connections: int) -> neat.genome.Genome:
        """ Creates a new random genome.

        Args:
            num_inputs (int): Number of input nodes in the new genome.
            num_outputs (int): Number of output nodes in the new genome.
            config (Config): Settings of the current evolution session.
            id_handler (IdHandler): ID handler used to assign IDs to the new
                genome's hidden nodes and connections.
            max_hidden_nodes (int): Maximum number of hidden nodes the new
                genome can have. The number of hidden nodes in the genome will
                be randomly picked from the interval `[0, max_hidden_nodes]`.
            max_hidden_connections (int): Maximum number of hidden connections
                (connections involving at least one hidden node) the new genome
                can have. The number of hidden connections in the genome will be
                randomly picked from the interval `[0, max_hidden_connections]`.

        Returns:
            The randomly generated genome.
        """
        new_genome = super(StaticConvGenome, cls).random_genome(
            num_inputs, num_outputs, config, id_handler,
            max_hidden_nodes, max_hidden_connections
        )

        # TODO
        return new_genome

    def process(self, X: Sequence[float]) -> np.array:
        """ TODO

        """
        return super().process(X)
