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

""" Implementation of the base abstract class that defines a population of
genomes, each of which encodes a neural network.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, List, Optional, Type, TypeVar

import numpy as np

import nevopy as ne
from nevopy.processing.base_scheduler import ProcessingScheduler

TGenome = TypeVar("TGenome", bound="ne.base_genome.BaseGenome")


class BasePopulation(ABC, Generic[TGenome]):
    """ Base abstract class that defines a population of genomes (neural nets).

    This base abstract class defines a population of genomes (each of which
    encodes a neural network) to be evolved through neuroevolution. It's in this
    class' subclasses where the core of `NEvoPy's` neuroevolutionary algorithms
    are implemented.

    Args:
        size (int): Number of genomes (constant) in the population.
        processing_scheduler (ProcessingScheduler): Processing scheduler to be
            used by the population.

    Attributes:
        scheduler (ProcessingScheduler): Processing scheduler used by the
            population. It's responsible for abstracting the details on how the
            processing is done (whether it's sequential or distributed, local or
            networked, etc).
        genomes (List[TGenome]): List with the genomes (neural networks being
            evolved) currently in the population.
        stop_evolving (bool): Flag that when set to `True` stops the
            evolutionary process being executed by the :meth:`.evolve` method.
    """

    #: Default processing scheduler to be used by the population.
    DEFAULT_SCHEDULER = None  # type: Optional[Type[ProcessingScheduler]]

    def __init__(self,
                 size: int,
                 processing_scheduler: ProcessingScheduler):
        self._size = size
        self.scheduler = processing_scheduler
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
               callbacks: Optional[List["ne.callbacks.Callback"]] = None,
               **kwargs) -> "ne.callbacks.History":
        """ Evolves the population of genomes through neuroevolution.

        This is the main method of this class. It's here where the main loop of
        the neuroevolutionary algorithm implemented is located.

        Args:
            generations (int): Maximum number of evolutionary generations.
            fitness_function (Callable[[TGenome], float]): Fitness function used
                to compute the fitness of a genome. It must take an instance of
                class:`.BaseGenome` as input and return the genome's fitness
                (float).
            callbacks (Optional[List["ne.callbacks.Callback"]]): List with
                instances of :class:`.Callback`. The callbacks will be called
                during different stages of an evolutionary generation. They can
                be used to customize the algorithm's behaviour.

        Returns:
            An instance of :class:`nevopy.callbacks.History` containing relevant
            information about the evolutionary session.
        """

    def fittest(self) -> TGenome:
        """ Returns the most fit genome in the population. """
        return self.genomes[int(np.argmax([g.fitness for g in self.genomes]))]

    def average_fitness(self) -> float:
        """ Returns the average fitness of the population's genomes. """
        return np.mean([g.fitness for g in self.genomes])

    def save(self, abs_path: str) -> None:
        """ Saves the population on the absolute path provided.

        This method uses, by default, :py:mod:`pickle` to save the population.
        The processing scheduler used by the population won't be saved (a new
        one will have to be assigned to the population when it's loaded again).

        Args:
            abs_path (str): Absolute path of the saving file. If the given path
                doesn't end with the suffix ".pkl", it will be automatically
                added to it.
        """
        scheduler_cache = self.scheduler
        self.scheduler = None
        ne.utils.pickle_save(self, abs_path)
        self.scheduler = scheduler_cache

    @classmethod
    def load(cls,
             abs_path: str,
             scheduler: Optional[ProcessingScheduler] = None,
    ) -> "BasePopulation":
        """ Loads the population from the given absolute path.

        This method uses, by default, :py:mod:`pickle` to load the population.

        Args:
            abs_path (str): Absolute path of the saved ".pkl" file.
            scheduler (Optional[ProcessingScheduler]): Processing scheduler to
                be used by the population. If `None`, the default one will be
                used.

        Returns:
            The loaded population.
        """
        pop = ne.utils.pickle_load(abs_path)
        # pylint: disable=not-callable
        pop.scheduler = (scheduler if scheduler is not None
                         else cls.DEFAULT_SCHEDULER())
        return pop
