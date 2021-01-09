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

import ray
from nevopy.processing.base_scheduler import *
from typing import List


class RayProcessingScheduler(ProcessingScheduler):
    """
    TODO
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
        """ TODO

        Args:
            items:
            func:

        Returns:

        """
        func_id = ray.put(func)
        return ray.get([_func_wrapper.remote(item, func_id)
                        for item in items])


@ray.remote
def _func_wrapper(item, func):
    """ TODO

    Returns:

    """
    return func(item)