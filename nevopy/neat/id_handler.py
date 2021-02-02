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

""" This module implements the ID handler, use to assign IDs to species,
genomes, hidden nodes and connections. In the case of nodes and connections
genes, the ID can also be interpreted as an innovation number.
"""

from typing import Dict


class IdHandler:
    """ Handles the assignment of IDs.

    An ID handler manages the assignment of IDs to species, genomes, hidden
    nodes and connections. In the case of nodes and connections genes, the ID
    can also be interpreted as an innovation number.

    "The innovation numbers are historical markers that identify the original
    historical ancestor of each gene. New genes are assigned new increasingly
    higher numbers." - :cite:`stanley:ec02`

    The ID handler implements the following solution:

    "A possible problem is that the same structural innovation will receive
    different innovation numbers in the same generation if it occurs by chance
    more than once. However, by keeping a list of the innovations that occurred
    in the current generation, it is possible to ensure that when the same
    structure arises more than once through independent mutations in the same
    generation, each identical mutation is assigned the same innovation number.
    Thus, there is no resultant explosion of innovation numbers." -
    :cite:`stanley:ec02`

    In `NEvoPy`, it's possible to configure the rate at which innovation numbers
    are reset (see :attr:`.NeatConfig.reset_innovations_period`).

    Warning:
        This class isn't compatible with parallel processing.

    Args:
        num_inputs (int): Number of input nodes in the genomes.
        num_outputs (int): Number of output nodes in the genomes.
        has_bias (bool): Whether the genomes have a bias node.
    """

    def __init__(self,
                 num_inputs: int,
                 num_outputs: int,
                 has_bias: bool) -> None:
        self._node_counter = num_inputs + num_outputs + 1 if has_bias else 0
        self._connection_counter = num_inputs * num_outputs
        self._species_counter = 0
        self._new_connections_ids = {}  # type: Dict[int, Dict[int, int]]
        self._new_nodes_ids = {}        # type: Dict[int, Dict[int, int]]
        self.reset_counter = 0

    def reset(self) -> None:
        """ Resets the cache of new nodes and connections.

        This resets the handler's cached innovations.
        """
        self._new_connections_ids = {}
        self._new_nodes_ids = {}
        self.reset_counter = 0

    def next_species_id(self):
        """ Returns a new unique ID for a species. """
        sid = self._species_counter
        self._species_counter += 1
        return sid

    def next_hidden_node_id(self, src_id: int, dest_id: int) -> int:
        """ Returns an ID / innovation number for a hidden node.

        A hidden node is created by breaking an existing connection of the
        genome in two. Consider two nodes `A` and `B`, both of which are present
        in multiple genomes of the population. While the ID handler isn't reset,
        hidden nodes created by breaking the connection `A->B` will be assigned
        the same ID / innovation number.

        Args:
            src_id (int): ID of the source node of the connection being broken
                to create the new hidden node.
            dest_id (int): ID of the destination node of the connection being
                broken to create the new hidden node.

        Returns:
            An ID for the new hidden node.
        """
        if src_id is None or dest_id is None:
            raise RuntimeError("Trying to generate an ID to a node whose "
                               "parents (one or both) have \"None\" IDs!")

        if src_id in self._new_nodes_ids:
            if dest_id in self._new_nodes_ids[src_id]:
                return self._new_nodes_ids[src_id][dest_id]
        else:
            self._new_nodes_ids[src_id] = {}

        hid = self._node_counter
        self._node_counter += 1
        self._new_nodes_ids[src_id][dest_id] = hid
        return hid

    def next_connection_id(self, src_id: int, dest_id: int) -> int:
        """ Returns an ID / innovation number for a connection gene.

        The new connection is identified through the IDs of its source and
        destination nodes. While the ID handler isn't reset, connections that
        have the same source and destination nodes will be assigned the same ID.

        Args:
            src_id (int): ID of the new connection's source node.
            dest_id (int): ID of the new connection's destination node.

        Returns:
            An ID for the new connection.
        """
        if src_id in self._new_connections_ids:
            if dest_id in self._new_connections_ids[src_id]:
                return self._new_connections_ids[src_id][dest_id]
        else:
            self._new_connections_ids[src_id] = {}

        cid = self._connection_counter
        self._connection_counter += 1
        self._new_connections_ids[src_id][dest_id] = cid
        return cid
