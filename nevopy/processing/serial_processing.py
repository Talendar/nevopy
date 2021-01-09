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

""" Implements a simple wrapper for the serial processing of items.
"""

from nevopy.processing.base_scheduler import *
from typing import List


class SerialProcessingScheduler(ProcessingScheduler):
    """ Simple wrapper for the serial processing of items.

    This is scheduler is just a wrapper for the serial processing of items,
    i.e., the processing of one item at a time. It doesn't involve any explicit
    parallel processing.
    """

    def run(self,
            items: Iterable[T],
            func: ItemProcessingCallback
    ) -> List[R]:
        """ Sequentially processes the input items.

        Args:
            items (Iterable[T]): Iterable containing the items to be processed.
            func (ItemProcessingCallback): Callable (usually a function) that
                takes one item `T` as input and returns a result `R`. Generally,
                `T` is an individual in the population and `R` is the
                individual's fitness. If additional arguments must be passed to
                the callable you want to use, it's possible to use Python's
                :mod:`functools.partial` or to just wrap it with a function.

        Returns:
            A list containing the results of the processed items, in the order
            they are processed. This means that if the argument passed to
            `items` is a `Sequence`, the order of the results will match the
            order of the sequence.
        """
        return [func(item) for item in items]
