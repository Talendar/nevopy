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

This module contains a base model for a processing scheduler, the entity
responsible for managing the computation of a population's fitness in `NEvoPy's`
algorithms. Schedulers allow the implementation of the computation methods (like
the use of serial or parallel processing) to be separated from the
implementation of the neuroevolutionary algorithms.

Attributes:
    TProcItem (TypeVar): :py:class:`TypeVar` indicating an item to be scheduled
        for processing by a :class:`ProcessingScheduler`. Alias for
        TypeVar("TProcItem").
    TProcResult (TypeVar): :py:class:`TypeVar` indicating the result of
        processing a :attr:`TProcItem`. Alias for TypeVar("TProcResult").
"""

from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Sequence, TypeVar

TProcItem = TypeVar("TProcItem")
TProcResult = TypeVar("TProcResult")


class ProcessingScheduler(ABC):
    """ Defines a common interface for processing schedulers.

    In `NEvoPy`, a processing scheduler is responsible for managing the
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
            items: Sequence[TProcItem],
            func: Optional[Callable[[TProcItem], TProcResult]],
    ) -> List[TProcResult]:
        """ Processes the given items and returns a result.

        Main function of the scheduler. Call it to make the scheduler manage the
        processing of a batch of items.

        Args:
            items (Sequence[TProcItem]): Iterable containing the items to be
                processed.
            func (Optional[Callable[[TProcItem], TProcResult]]): Callable
                (usually a function) that takes one item :attr:`TProcItem` as
                input and returns a result :attr:`TProcResult` as output.
                Generally, :attr:`TProcItem` is an individual in the population
                and :attr:`TProcResult` is the individual's fitness. Since some
                scenarios requires the fitness of the population's individuals
                to be calculated together, at once, the use of this parameter is
                not mandatory (this decision is a implementation particularity
                of each sub-classed scheduler). If additional arguments must be
                passed to the callable you want to use, it's possible to use
                Python's :mod:`functools.partial`  or to just wrap it with a
                simple function.

        Returns:
            A list containing the results of the processing of each item. It is
            guaranteed that the ordering of the items in the returned list
            follows the order in which the items are yielded by the iterable
            passed as argument.
        """
        raise NotImplementedError()
