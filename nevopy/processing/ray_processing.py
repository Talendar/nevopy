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

import os
import ray
from nevopy.processing.base_scheduler import (ProcessingScheduler,
                                              TProcItem, TProcResult)
from typing import List, Optional, Sequence, Callable


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
            each raylet. By default, this is set based on virtual cores (value
            returned by os.cpu_count()).
        num_gpus (Optional[int]): Number of GPUs the user wishes to assign to
            each raylet. By default, this is set based on detected GPUs. If you
            are using TensorFlow, it's recommended for you to execute the
            following piece of code before importing the module:

                .. code-block:: python

                    import os
                    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

            This will prevent individual TensorFlow's sessions from allocating
            the entire GPU memory available.

        worker_gpu_frac (Optional[float]): Minimum fraction of a GPU a worker
            needs in order to use it. If there isn't enough GPU resources
            available for a worker when a task is assigned to it, it will not
            use any GPU resources. Here we consider the number of workers as
            being equal to the number of virtual CPU cores available. By
            default, this fraction is set to ``num_gpus / num_cpus``, which
            means that all workers will use the GPUs, each being able to access
            an equal fraction of them. Note that this might be a source of `out
            of memory errors`, since the GPU fraction assigned to each worker
            might be too low. It's usually better to manually select a fraction.
        **kwargs: Optional named arguments to be passed to `ray.init()`. For a
            complete list of the parameters of `ray.init()`, check `ray's`
            official docs (https://docs.ray.io/en/master/package-ref.html).
    """

    def __init__(self,
                 address: Optional[str] = None,
                 num_cpus: Optional[int] = None,
                 num_gpus: Optional[int] = None,
                 worker_gpu_frac: Optional[float] = None,
                 **kwargs) -> None:
        if ray.is_initialized():
            RuntimeError("An existing ray runtime was detected! Stop it before "
                         "instantiating this class to avoid conflicts.")

        self._num_cpus = num_cpus if num_cpus is not None else os.cpu_count()
        ray.init(address=address,
                 num_cpus=self._num_cpus,
                 num_gpus=num_gpus,
                 **kwargs)
        self._num_gpus = ray.available_resources()["GPU"]
        self._worker_gpu_frac = (worker_gpu_frac if worker_gpu_frac is not None
                                 else self._num_gpus / self._num_cpus)

        print(f"Ray's Resources: CPUs: {self._num_cpus}  |  "
              f"GPUs: {self._num_gpus}  |  "
              f"GPU frac: {self._worker_gpu_frac:.4f}")

    def run(self,
            items: Sequence[TProcItem],
            func: Callable[[TProcItem], TProcResult],
    ) -> List[TProcResult]:
        """ Processes the given items and returns a result.

        Main function of the scheduler. Call it to make the scheduler manage the
        parallel processing of a batch of items using `ray`.

        Args:
            items (Sequence[TProcItem]): Iterable containing the items to be
                processed.
            func (Callable[[TProcItem], TProcResult]): Callable (usually a
                function) that takes one item :attr:`.TProcItem` as input and
                returns a result :attr:`.TProcResult` as output. Generally,
                :attr:`.TProcItem` is an individual in the population and
                :attr:`.TProcResult` is the individual's fitness. If additional
                arguments must be passed to the callable you want to use, it's
                possible to use Python's :mod:`functools.partial` or to just
                wrap it with a simple function. The callable doesn't need to be
                annotated with `ray.remote`, this is handled for you.

        Returns:
            A list containing the results of the processing of each item. It is
            guaranteed that the ordering of the items in the returned list
            follows the order in which the items are yielded by the iterable
            passed as argument. This means that if the argument  passed to
            `items` is a :py:class:`Sequence`, the order of the results will
            match the order of the sequence.
        """
        func_id = ray.put(func)
        if self._num_gpus == 0:
            return ray.get([_func_wrapper.remote(func_id, item)
                            for item in items])

        NUM_WORKERS = self._num_cpus
        processing_refs = []
        results_dict = {}
        gpu_processing_refs = set()
        ref2idx = {}

        gpu_available = self._num_gpus
        idx = 0
        while len(results_dict) != len(items):
            # retrieving results
            if len(processing_refs) > 0:
                done_refs, processing_refs = ray.wait(processing_refs)
                for ref, result in zip(done_refs, ray.get(done_refs)):
                    if ref in gpu_processing_refs:
                        gpu_available += self._worker_gpu_frac
                        gpu_processing_refs.remove(ref)
                    results_dict[ref2idx[ref]] = result

            # assigning work
            while len(processing_refs) < NUM_WORKERS and idx < len(items):
                if gpu_available >= self._worker_gpu_frac:
                    gpu_available -= self._worker_gpu_frac
                    ref = _func_wrapper.options(num_gpus=self._worker_gpu_frac)\
                                       .remote(func_id, items[idx])
                    gpu_processing_refs.add(ref)
                else:
                    ref = _func_wrapper.options(num_gpus=0).remote(func_id,
                                                                   items[idx])
                processing_refs.append(ref)
                ref2idx[ref] = idx
                idx += 1

        # returning results
        assert len(items) == len(results_dict)
        return [results_dict[i] for i in sorted(results_dict)]


@ray.remote
def _func_wrapper(func: Callable[[TProcItem], TProcResult],
                  item: TProcItem) -> TProcResult:
    """ Wrapper function to be used by `ray`. """
    # os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    return func(item)
