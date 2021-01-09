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

""" Implements a processing scheduler that uses the `ray` framework.

By using `ray` (https://github.com/ray-project/ray), the scheduler is able to
implement parallel processing, either on a single machine or on a cluster.
"""

import ray
from nevopy.processing.base_scheduler import (ProcessingScheduler,
                                              ItemProcessingCallback, T, R)
from typing import List, Optional, Iterable


class RayProcessingScheduler(ProcessingScheduler):
    """ Scheduler that uses `ray` to implement parallel processing.

    Ray is an open source framework that provides a simple, universal API for
    building distributed applications. This scheduler uses it to implement
    parallel processing. It's possible to either run `ray` on a single machine
    or on a cluster. For more information about the `ray` framework, checkout
    the project's GitHub page: https://github.com/ray-project/ray.

    It's possible to view the `ray's` dashboard at http://127.0.0.1:8265. It
    contains useful information about the distribution of work and usage of
    resources by `ray`.

    When this class is instantiated, a new `ray` runtime is created. You should
    close existing `ray` runtimes before creating a new `ray` scheduler, to
    avoid possible conflicts. If, for some reason, you want to use a currently
    running `ray` runtime instead of creating a new one, pass `True` as argument
    to `ignore_reinit_error`.

    This class is, basically, a simple `wrapper` for `ray`. If you're an
    advanced user and this scheduler doesn't meet your needs, it's recommended
    that you implement your own scheduler by inheriting
    :class:`.ProcessingScheduler`.

    Args:
        address (Optional[str]): The address of the Ray cluster to connect to.
            If this address is not provided, then this command will start Redis,
            a raylet, a plasma store, a plasma manager, and some workers.
            It will also kill these processes when Python exits. If the driver
            is running on a node in a Ray cluster, using `auto` as the value
            tells the driver to detect the the cluster, removing the need to
            specify a specific node address.
        num_cpus (Optional[int]): Number of CPUs the user wishes to assign to
            each raylet. By default, this is set based on virtual cores.
        **kwargs: Optional named arguments to be passed to `ray.init()`. For a
            complete list of the parameters of `ray.init()`, check `ray's`
            official docs (https://docs.ray.io/en/master/package-ref.html).
    """

    def __init__(self,
                 address: Optional[str] = None,
                 num_cpus: Optional[int] = None,
                 **kwargs) -> None:
        if ray.is_initialized():
            RuntimeError("An existing ray runtime was detected! Stop it before "
                         "instantiating this class to avoid conflicts.")
        ray.init(address=address, num_cpus=num_cpus, **kwargs)

    def run(self,
            items: Iterable[T],
            func: ItemProcessingCallback,
    ) -> List[R]:
        """ Processes the given items and returns a result.

        Main function of the scheduler. Call it to make the scheduler manage the
        parallel processing of a batch of items using `ray`.

        Args:
            items (Iterable[T]): Iterable containing the items to be processed.
            func (Optional[ItemProcessingCallback]): Callable (usually a
                function) that takes one item `T` as input and returns a result
                `R`. Generally, `T` is an individual in the population and `R`
                is the individual's fitness. If additional arguments must be
                passed to the callable you want to use, it's possible to use
                Python's :mod:`functools.partial` or to just wrap it with a
                function. The function don't need to be annotated with
                `ray.remote`, this is handled for you.

        Returns:
            A list containing the results of the processing of each item, in the
            order they are processed. This means that if the argument passed to
            `items` is a `Sequence`, the order of the results will match the
            order of the sequence.
        """
        func_id = ray.put(func)
        return ray.get([_func_wrapper.remote(item, func_id)
                        for item in items])


@ray.remote
def _func_wrapper(item: T, func: ItemProcessingCallback) -> R:
    """ Wrapper function to be used by `ray`. """
    return func(item)
