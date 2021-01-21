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
either directly or indirectly. This module declares the base abstract class that
must be inherited by all the different classes of genomes used by the
neuroevolutionary algorithms in `NEvoPY`.
"""

from abc import ABC, abstractmethod
from typing import Any

import pickle
from pathlib import Path


class BaseGenome(ABC):
    """ Defines the general behaviour of a genome in `NEvoPY`.

    This class must be inherited by all the different classes of genomes present
    in `NEvoPY`

    In the context of neuroevolution, a genome is the entity subject to the
    evolutionary process. It encodes a neural network (the genome's phenotype),
    either directly or indirectly.

    As pointed out by :cite:`stanley:ec02`, direct encoding schemes, employed in
    most cases, specify in the genome every connection and node that will appear
    in the phenotype. In contrast, indirect encodings usually only specify rules
    for constructing a phenotype. These rules can be layer specifications or
    growth rules through cell division.

    One of the goals of this base abstract class is to abstract these details
    for the user, defining a general interface for different types of genomes
    used by the different neuroevolution algorithms in `NEvoPY`. Generally, for
    `NEvoPY`, there is no distinction between a genome and the network it
    encodes.

    A genome must be capable of processing inputs based on its nodes and
    connections in order to produce an output, emulating a neural network. It
    also must be able to mutate and to generate offspring, in order to evolve.

    Attributes:
        fitness (float): The current fitness of the genome.
    """

    def __init__(self) -> None:
        self.fitness = 0.0

    @abstractmethod
    def process(self, X: Any) -> Any:
        """ Feeds the given input to the neural network encoded by the genome.

        Args:
            X (Any): The input(s) to be fed to the neural network encoded by the
                genome. Usually a `NumPy ndarray` or a `TensorFlow tensor`.

        Returns:
            The output of the network. Usually a `NumPy ndarray` or a
            `TensorFlow tensor`.

        Raises:
            InvalidInputError: If the shape of ``X`` doesn't match the input
                shape expected by the network.
        """

    def __call__(self, X: Any) -> Any:
        """ Wraps a call to :meth:`.process`. """
        return self.process(X)

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

    # todo:
    # def shallow_copy(self):
    #     pass

    @abstractmethod
    def deep_copy(self) -> "BaseGenome":
        """ Makes an exact/deep copy of the genome.

        Returns:
            An exact/deep copy of the genome.
        """

    @abstractmethod
    def mate(self, other: Any) -> "BaseGenome":
        """ Mates two genomes to produce a new genome (offspring).

        Implements the sexual reproduction between a pair of genomes. The new
        genome inherits information from both parents (not necessarily in an
        equal proportion)

        Args:
            other (Any): The second genome . If it's not compatible for mating
                with the current genome (`self`), an exception will be raised.

        Returns:
            A new genome (the offspring born from the sexual reproduction
            between the current genome and the genome passed as argument.

        Raises:
            IncompatibleGenomesError: If the genome passed as argument to
                ``other`` is incompatible with the current genome (`self`).
        """

    def save(self, abs_path: str) -> None:
        """ Saves the genome to the given absolute path.

        This method uses, by default, :py:mod:`pickle` to save the genome.

        Args:
            abs_path (str): Absolute path of the saving file. If the given path
                doesn't end with the suffix ".pkl", it will be automatically
                added.
        """
        p = Path(abs_path)
        if not p.suffixes:
            p = Path(str(abs_path) + ".pkl")
        p.parent.mkdir(parents=True, exist_ok=True)

        with open(str(p), "wb") as out_file:
            pickle.dump(self, out_file, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, abs_path: str) -> "BaseGenome":
        """ Loads the genome from the given absolute path.

        This method uses, by default, :py:mod:`pickle` to load the genome.

        Args:
            abs_path (str): Absolute path of the saved ".pkl" file.

        Returns:
            The loaded genome.
        """
        with open(abs_path, "rb") as in_file:
            return pickle.load(in_file)


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