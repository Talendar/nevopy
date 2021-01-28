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

from typing import Any, Optional, Callable, List, TypeVar, Generic
from abc import ABC, abstractmethod
import numpy as np
import nevopy


TGenome = TypeVar("TGenome", bound="nevopy.base_genome.BaseGenome")


class Population(ABC, Generic[TGenome]):
    """
    TODO
    """

    def __init__(self,
                 size: int):
        self._size = size
        self.genomes = []  # type: List[TGenome]
        self.stop_evolving = False

    @property
    def size(self) -> int:
        """ Size of the population. """
        return self._size

    @property
    @abstractmethod
    def config(self) -> Any:
        """ Config object that stores the settings used by the population. """

    @abstractmethod
    def evolve(self,
               generations: int,
               fitness_function: Callable[[TGenome], float],
               callbacks: Optional[List["nevopy.callbacks.Callback"]] = None,
               **kwargs) -> "nevopy.callbacks.History":
        """

        Args:
            generations:
            fitness_function:
            callbacks:

        Returns:

        """

    def fittest(self) -> TGenome:
        """ Returns the most fit genome in the population. """
        return self.genomes[int(np.argmax([g.fitness for g in self.genomes]))]

    def average_fitness(self) -> float:
        """ Returns the average fitness of the population's genomes. """
        return np.mean([g.fitness for g in self.genomes])
