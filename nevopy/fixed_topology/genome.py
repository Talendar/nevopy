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

from typing import Any, List
from nevopy.base_genome import BaseGenome, IncompatibleGenomesError
from nevopy.fixed_topology.layers import BaseLayer


class FixedTopologyGenome(BaseGenome):
    """
    TODO
    """

    def __init__(self):
        super().__init__()
        self.layers = []  # type: List[BaseLayer]

    def process(self, X: Any) -> Any:
        pass

    def reset(self) -> None:
        """ This method doesn't do anything.

        In this implementation, the default fixed-topology networks do not need
        to reset any of its internal states before the start of a new
        generation.
        """
        pass

    def mutate_weights(self) -> None:
        """ Randomly mutates the weights of the genome's connections.

        TODO
        """

    def deep_copy(self) -> "FixedTopologyGenome":
        """ TODO

        Returns:

        """

    def mate(self, other: "FixedTopologyGenome") -> "FixedTopologyGenome":
        """ TODO

        Args:
            other:

        Returns:

        """
