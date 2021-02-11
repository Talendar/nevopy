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

""" Implementation of the :class:`.NeatSpecies` class.
"""

from typing import List, Optional

import numpy as np

from nevopy.neat.genomes import NeatGenome


class NeatSpecies:
    """ Represents a species within NEAT's evolutionary environment.

    Args:
        species_id (int): Unique identifier of the species.
        generation (int): Current generation. The generation in which the
            species is born.

    Attributes:
        representative (Optional[NeatGenome]): Genome used to represent the
            species.
        members (List[NeatGenome]): List with the genomes that belong to the
            species.
        last_improvement (int): Generation in which the species last showed
            improvement of its fitness. The species fitness in a given
            generation is equal to the fitness of the species most fit genome on
            that generation.
        best_fitness (Optional[float]): The last calculated fitness of the
            species most fit genome.
    """

    def __init__(self, species_id: int, generation: int) -> None:
        self._id = species_id
        self.representative = None  # type: Optional[NeatGenome]
        self.members = []           # type: List[NeatGenome]

        self._creation_gen = generation
        self.last_improvement = generation
        self.best_fitness = None   # type: Optional[float]

    @property
    def id(self) -> int:
        """ Unique identifier of the species. """
        return self._id

    def random_representative(self) -> None:
        """ Randomly chooses a new representative for the species. """
        self.representative = np.random.choice(self.members)

    def avg_fitness(self) -> float:
        """ Returns the average fitness of the species genomes. """
        return float(np.mean([g.fitness for g in self.members]))

    def fittest(self) -> NeatGenome:
        """ Returns the fittest member of the species. """
        return self.members[int(np.argmax([g.fitness for g in self.members]))]
