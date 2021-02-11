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

""" Declares the base abstract class that defines the behaviour of a genome.

In the context of neuroevolution, a genome is the entity subject to the
evolutionary process. It encodes a neural network (the genome's phenotype),
either directly or indirectly.

This module declares the base abstract class that must be inherited by all the
different classes of genomes used by the neuroevolutionary algorithms in
`NEvoPy`.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Union

from nevopy.utils.utils import pickle_load
from nevopy.utils.utils import pickle_save


class BaseGenome(ABC):
    """ Defines the general behaviour of a genome in `NEvoPy`.

    This class must be inherited by all the different classes of genomes present
    in `NEvoPy`

    In the context of neuroevolution, a genome is the entity subject to the
    evolutionary process. It encodes a neural network (the genome's phenotype),
    either directly or indirectly.

    As pointed out by :cite:`stanley:ec02`, direct encoding schemes, employed in
    most cases, specify in the genome every connection and node that will appear
    in the phenotype. In contrast, indirect encodings usually only specify rules
    for constructing a phenotype. These rules can be layer specifications or
    growth rules through cell division.

    One of the goals of this base abstract class is to abstract those details
    for the user, defining a general interface for the different types of
    genomes used by the different neuroevolutionary algorithms in `NEvoPy`.
    Generally, for `NEvoPy`, there is no distinction between a genome and the
    network it encodes.

    A genome must be capable of processing inputs based on its nodes and
    connections in order to produce an output, emulating a neural network. It
    also must be able to mutate and to generate offspring, in order to evolve.

    Attributes:
        fitness (float): The current fitness value of the genome.
    """

    def __init__(self) -> None:
        self.fitness = 0.0

    @property
    @abstractmethod
    def input_shape(self) -> Optional[Union[int, Tuple[int, ...]]]:
        """ The expected shape of the inputs that will be fed to the genome.

        Returns:

            * ``None``, if an input shape has not been defined yet;
            * An ``int``, if the expected inputs are one-dimensional;
            * A ``tuple`` with the expected inputs' dimensions, if they're
              multi-dimensional.

        """

    @property
    @abstractmethod
    def config(self) -> Any:
        """ Settings of the current evolutionary session.

        If `None`, a config object hasn't been assigned to this genome yet.
        """

    @config.setter
    def config(self, c) -> None:
        """ Sets the config object of the genome. """

    @abstractmethod
    def process(self, x: Any) -> Any:
        """ Feeds the given input to the neural network encoded by the genome.

        Args:
            x (Any): The input(s) to be fed to the neural network encoded by the
                genome. Usually a `NumPy ndarray` or a `TensorFlow tensor`.

        Returns:
            The output of the network. Usually a `NumPy ndarray` or a
            `TensorFlow tensor`.

        Raises:
            InvalidInputError: If the shape of ``X`` doesn't match the input
                shape expected by the network.
        """

    def __call__(self, x: Any) -> Any:
        """ Wraps a call to :meth:`.process`. """
        return self.process(x)

    @abstractmethod
    def reset(self) -> None:
        """ Prepares the genome for a new generation.

        In this method, relevant actions related to the reset of a genome's
        internal state, in order to prepare it to a new generation, are
        implemented. The implementation of this method is not mandatory.
        """

    @abstractmethod
    def mutate_weights(self) -> None:
        """ Randomly mutates the weights of the genome's connections. """

    @abstractmethod
    def random_copy(self) -> "BaseGenome":
        """ Makes a deep copy of the genome, but with random weights.

        Returns:
            A deep copy of the genome with the same topology of the original
            genome, but random connections weights.
        """

    @abstractmethod
    def deep_copy(self) -> "BaseGenome":
        """ Makes an exact/deep copy of the genome.

        Returns:
            An exact/deep copy of the genome. It has the same topology and
            connections weights of the original genome.
        """

    @abstractmethod
    def mate(self, other: Any) -> "BaseGenome":
        """ Mates two genomes to produce a new genome (offspring).

        Implements the sexual reproduction between a pair of genomes. The new
        genome inherits information from both parents (not necessarily in an
        equal proportion)

        Args:
            other (Any): The second genome. If it's not compatible for mating
                with the current genome (`self`), an exception will be raised.

        Returns:
            A new genome (the offspring born from the sexual reproduction
            between the current genome and the genome passed as argument).

        Raises:
            IncompatibleGenomesError: If the genome passed as argument to
                ``other`` is incompatible with the current genome (`self`).
        """

    @abstractmethod
    def distance(self, other: Any) -> float:
        """ Calculates the distance between two genomes.

        Args:
            other (BaseGenome): The other genome.

        Returns:
            A float representing the distance between the two genomes. The lower
            the distance, the more similar the two genomes are.
        """

    @abstractmethod
    def visualize(self, **kwargs) -> None:
        """ Utility method for visualizing the genome's neural network. """

    def save(self, abs_path: str) -> None:
        """ Saves the genome to the given absolute path.

        This method uses, by default, :py:mod:`pickle` to save the genome.

        Args:
            abs_path (str): Absolute path of the saving file. If the given path
                doesn't end with the suffix ".pkl", it will be automatically
                added.
        """
        pickle_save(self, abs_path)

    @classmethod
    def load(cls, abs_path: str) -> "BaseGenome":
        """ Loads the genome from the given absolute path.

        This method uses, by default, :py:mod:`pickle` to load the genome.

        Args:
            abs_path (str): Absolute path of the saved ".pkl" file.

        Returns:
            The loaded genome.
        """
        return pickle_load(abs_path)


class InvalidInputError(Exception):
    """
    Indicates that the a given input isn't compatible with a given neural
    network.
    """


class IncompatibleGenomesError(Exception):
    """
    Indicates that an attempt has been made to mate (sexual reproduction) two
    incompatible genomes.
    """
