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

""" Defines the abstract class that serves as a base for all the neural layers
used by fixed-topology neuroevolutionary algorithms.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

from numpy import ndarray

from nevopy.genetic_algorithm.config import GeneticAlgorithmConfig
from nevopy.utils import pickle_load
from nevopy.utils import pickle_save


class BaseLayer(ABC):
    """ Abstract base class that defines a neural network layer.

    This abstract base class defines the general structure and behaviour of a
    fixed topology neural network layer in the context of neuroevolutionary
    algorithms.

    Args:
        config (Optional[FixedTopologyConfig]): Settings being used in the
            current evolutionary session. If `None`, a config object must be
            assigned to the layer later on, before calling the methods that
            require it.
        input_shape (Optional[Tuple[int, ...]]): Shape of the data that will be
            processed by the layer. If `None`, an input shape for the layer must
            be manually specified later or be inferred from an input sample.
        mutable (Optional[bool]): Whether or not the layer can have its weights
            changed (mutation).

    Attributes:
        config (Optional[FixedTopologyConfig]): Settings being used in the
            current evolutionary session. If `None`, a config object hasn't been
            assigned to the layer yet.
        mutable (bool): Whether or not the layer can have its weights changed
            (mutation).
    """

    def __init__(self,
                 config: Optional[GeneticAlgorithmConfig] = None,
                 input_shape: Optional[Tuple[int, ...]] = None,
                 mutable: Optional[bool] = True) -> None:
        self.config = config
        self._input_shape = input_shape
        self.mutable = mutable

    @property
    @abstractmethod
    def weights(self) -> List[ndarray]:
        """ A list with the layer's weight matrices as numpy arrays.

        For most layers, it's a tuple containing a weight matrix and a bias
        vector.
        """

    @weights.setter
    def weights(self, new_weights: List[ndarray]) -> None:
        """ Setter for the :meth:`.weights` property. """

    @property
    def input_shape(self) -> Optional[Tuple[int, ...]]:
        """ The expected shape of an input for the layer.

        Returns:
            A tuple with the layer's input shape or `None` if an input shape
            hasn't been specified yet.
        """
        return self._input_shape

    @abstractmethod
    def build(self, input_shape: Tuple[int, ...]) -> None:
        """ Builds the layer's weight and bias matrices.

        If the layer has already been built, it will be built again (new weight
        and bias matrices will be generated).

        Args:
            input_shape (Tuple[int, ...]): Tuple with the shape of the inputs
                that will be fed to the layer.

        Raises:
            ValueError: If the layer isn't compatible with the given input
                shape.
        """

    @abstractmethod
    def process(self, x: Any) -> Any:
        """ Feeds the given input(s) to the layer.

        This is where the layer's logic lives. If the layer hasn't been built
        yet, it will be automatically built using the given input shape.

        Args:
            x (Any): The input(s) to be fed to the layer. Usually a
                `NumPy ndarray` or a `TensorFlow tensor`.

        Returns:
            The output of the layer. Usually a `NumPy ndarray` or a
            `TensorFlow tensor`.

        Raises:
            InvalidInputError: If the shape of ``x`` doesn't match the input
                shape expected by the layer.
        """

    def __call__(self, x: Any) -> Any:
        """ Wraps a call to :meth:`.process`. """
        return self.process(x)

    @abstractmethod
    def random_copy(self) -> "BaseLayer":
        """ Makes a random copy of the layer.

        Returns:
            A new layer with the same topology of the current layer, but with
            newly initialized weights and biases. If the layer is immutable, a
            deep copy (:meth:`.BaseLayer.deep_copy`) of the layer is returned
            instead.
        """

    @abstractmethod
    def deep_copy(self) -> "BaseLayer":
        """ Makes an exact/deep copy of the layer.

        Returns:
            An exact/deep copy of the layer, including its weights and biases.
        """

    @abstractmethod
    def mutate_weights(self) -> None:
        """ Randomly mutates the weights of the layer's connections.

        If the layer is immutable, this method doesn't do anything.
        """

    @abstractmethod
    def mate(self, other: Any) -> "BaseLayer":
        """ Mates two layers to produce a new layer (offspring).

        Implements the sexual reproduction between a pair of layers. The new
        layer inherits information from both parents (not necessarily in an
        equal proportion)

        Args:
            other (Any): The second layer . If it's not compatible for mating
                with the current layer (`self`), an exception will be raised.

        Returns:
            A new layer (the offspring born from the sexual reproduction between
            the current layer and the layer passed as argument. If the layer is
            immutable, ``other`` is expected to be equal to ``self``, so a deep
            copy (:meth:`.BaseLayer.deep_copy`) of the layer is returned.

        Raises:
            IncompatibleLayersError: If the layer passed as argument to
                ``other`` is incompatible with the current layer (`self`).
        """

    def save(self, abs_path: str) -> None:
        """ Saves the layer on the absolute path provided.

        This method uses, by default, :py:mod:`pickle` to save the layer.

        Args:
            abs_path (str): Absolute path of the saving file. If the given path
                doesn't end with the suffix ".pkl", it will be automatically
                added to it.
        """
        pickle_save(self, abs_path)

    @classmethod
    def load(cls, abs_path: str) -> "BaseLayer":
        """ Loads the layer from the given absolute path.

        This method uses, by default, :py:mod:`pickle` to load the layer.

        Args:
            abs_path (str): Absolute path of the saved ".pkl" file. If the given
                path doesn't end with the suffix ".pkl", it will be
                automatically added to it.

        Returns:
            The loaded layer.
        """
        return pickle_load(abs_path)


class IncompatibleLayersError(Exception):
    """
    Indicates that an attempt has been made to mate (sexual reproduction) two
    incompatible layers.
    """
