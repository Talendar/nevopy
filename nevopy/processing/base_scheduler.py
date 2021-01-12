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

""" Defines a common interface for processing schedulers.

This module contains a base model for a process scheduler, the entity
responsible for managing the computation of the fitness of a population in
`nevopy's` algorithms. Schedulers allow the implementation of the computation
methods (like the use of serial or parallel processing) to be separated from the
implementation of the neuroevolutionary algorithms.
"""

from abc import ABC, abstractmethod
from typing import Iterable, TypeVar, Optional
from typing_extensions import Protocol

#: `TypeVar` indicating an item to be scheduled for processing.
T = TypeVar("T", contravariant=True)

#: `TypeVar` indicating the return value of processing of an item.
R = TypeVar("R", covariant=True)


class ItemProcessingCallback(Protocol[T, R]):
    """
    Defines an interface for a callback used by a processing scheduler to
    process an individual item. The type `T` is the item's type and the type `R`
    is the type of the result of processing the item.
    """

    def __call__(self, item: T) -> R:
        """ Defines the expected signature of the callback.

        Args:
            item (T): An item to be processed. In the context of neuroevolution,
                it's usually an individual whose fitness we want to compute.

        Returns:
            An expected result. In the context of neuroevolution, it's usually
            the fitness of the individual.
        """
        raise NotImplementedError()


class ProcessingScheduler(ABC):
    """ Defines a common interface for processing schedulers.

    In `nevopy`, a processing scheduler is responsible for managing the
    computation of the fitness of a population of individuals being evolved.
    This abstract class defines a common interface for processing schedulers
    used by different algorithms. Schedulers allow the implementation of the
    computation methods (like the use of serial or parallel processing) to be
    separated from the implementation of the neuroevolutionary algorithms.

    Implementing your own processing scheduler is useful when you want to
    customize the computation of the population's fitness. You can, for example,
    implement a scheduler that makes use of multiple CPU cores or GPUs (parallel
    processing).
    """

    @abstractmethod
    def run(self,
            items: Iterable[T],
            func: Optional[ItemProcessingCallback],
    ) -> Iterable[R]:
        """ Processes the given items and returns a result.

        Main function of the scheduler. Call it to make the scheduler manage the
        processing of a batch of items.

        Args:
            items (Iterable[T]): Iterable containing the items to be processed.
            func (Optional[ItemProcessingCallback]): Callable (usually a
                function) that takes one item `T` as input and returns a result
                `R`. Generally, `T` is an individual in the population and `R`
                is the individual's fitness. Since some scenarios requires the
                fitness of the population's individuals to be calculated
                together, at once, this parameter is not always used. If
                additional arguments must be passed to the callable you want to
                use, it's possible to use Python's :mod:`functools.partial` or
                to just wrap it with a function.

        Returns:
            An iterable containing the results of the processing of each item.
            In some situations, it might be required for the order of the
            results to match the order of the items passed as input. In others,
            it's possible to map each item to its result. These decisions are
            implementation particularities of each scheduler.
        """
        raise NotImplementedError()
