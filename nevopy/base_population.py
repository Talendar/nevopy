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

    # def save(self, abs_path: str) -> None:
    #     """ Saves the population on the absolute path provided.
    #
    #     This method uses :py:mod:`pickle` to save the genome. The processing
    #     scheduler used by the population won't be saved (a new one will have to
    #     be assigned to the population when it's loaded again).
    #
    #     Args:
    #         abs_path (str): Absolute path of the saving file. If the given path
    #             doesn't end with the suffix ".pkl", it will be automatically
    #             added to it.
    #     """
    #     p = Path(abs_path)
    #     if not p.suffixes:
    #         p = Path(str(abs_path) + ".pkl")
    #     p.parent.mkdir(parents=True, exist_ok=True)
    #
    #     scheduler_cache = self._scheduler
    #     self._scheduler = None
    #     with open(str(p), "wb") as out_file:
    #         pickle.dump(self, out_file, pickle.HIGHEST_PROTOCOL)
    #
    #     self._scheduler = scheduler_cache
    #
    # @staticmethod
    # def load(abs_path: str,
    #          scheduler: Optional[ProcessingScheduler] = None,
    # ) -> "NeatPopulation":
    #     """ Loads the population from the given absolute path.
    #
    #     This method uses :py:mod:`pickle` to load the genome.
    #
    #     Args:
    #         abs_path (str): Absolute path of the saved ".pkl" file.
    #         scheduler (Optional[ProcessingScheduler]): Processing scheduler to
    #             be used by the population. If `None`, the default one will be
    #             used.
    #
    #     Returns:
    #         The loaded population.
    #     """
    #     with open(abs_path, "rb") as in_file:
    #         pop = pickle.load(in_file)
    #
    #     pop._scheduler = (scheduler if scheduler is not None
    #                       else NeatPopulation._DEFAULT_SCHEDULER())
    #     return pop
