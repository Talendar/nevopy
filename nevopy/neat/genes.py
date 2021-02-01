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

""" Implements the nodes (neurons) and edges (connections) of a genome.
"""

from enum import Enum
from typing import Callable, List, Optional, Tuple


class NodeGene:
    """ A gene that represents/encodes a neuron (node) in a neural network.

    A :class:`~NodeGene` is the portion of a :class:`.NeatGenome` that encodes a
    neuron (node) of the neural network encoded by the :class:`.NeatGenome`. It
    has an activation function, which is applied to inputs received from other
    nodes of the network.

    Args:
        node_id (int): The node's identifier / innovation number.
        node_type (NodeGene.Type): The node's type.
        activation_func (Callable[[float], float]): Activation function to be
            used by the node. It should receive a float as input and return a
            float (the resulting activation) as output.
        initial_activation (float): initial value of the node's activation (used
            when processing recurrent connections between nodes).

    Attributes:
        in_connections (List[ConnectionGene]): List with the connections
            (:class:`.ConnectionGene`) leaving this node, i.e., connections that
            have this node as the source.
        out_connections (List[ConnectionGene]): List with the connections
            (:class:`.ConnectionGene`) coming to this node, i.e., connections
            that have this node as the destination.
    """

    def __init__(self,
                 node_id: int,
                 node_type: "NodeGene.Type",
                 activation_func: Callable[[float], float],
                 initial_activation: float) -> None:
        assert node_id is not None
        self._id = node_id
        self._type = node_type
        self.initial_activation = initial_activation
        self._activation = initial_activation
        self.function = activation_func
        self.in_connections = []   # type: List[ConnectionGene]
        self.out_connections = []  # type: List[ConnectionGene]

    class Type(Enum):
        """ Specifies the possible types of node genes. """
        INPUT, BIAS, HIDDEN, OUTPUT = range(4)

    @property
    def id(self) -> int:
        """ Innovation ID of the gene.

        This ID is used to mate genomes and to calculate their difference.

        "The innovation numbers are historical markers that identify the
        original historical ancestor of each gene. New genes are assigned new
        increasingly higher numbers." - :cite:`stanley:ec02`
        """
        return self._id

    @property
    def type(self) -> "NodeGene.Type":
        """ Type of the node (input, bias, hidden or output). """
        return self._type

    @property
    def activation(self) -> float:
        """
        The node's cached activation value, i.e., the node's output when it was
        last processed.
        """
        return self._activation

    def activate(self, x: float) -> None:
        """ Applies the node's activation function to the given input.

        The node's activation value, i.e., the node's cached output, is updated
        by this call and can be later be accessed through the property
        :attr:`~NodeGene.activation`.

        Returns:
            None. The node's output is updated internally.
        """
        self._activation = self.function(x)

    def simple_copy(self) -> "NodeGene":
        """ Makes and returns a simple copy of this node.

        Wraps a call to this class' constructor.

        The copied node shares the same values for all the attributes of the
        source node, except for the connections. The copied node is created
        without any connections.

        Returns:
            A copy of this node without any connection.
        """
        return NodeGene(node_id=self._id,
                        node_type=self._type,
                        activation_func=self.function,
                        initial_activation=self.initial_activation)

    def reset_activation(self) -> None:
        """
        Resets the node's activation value (it's cached output) to its initial
        value.
        """
        self._activation = self.initial_activation


class ConnectionGene:
    """ A connection between two nodes.

    A connection gene represents/encodes a connection (edge) between two nodes
    (neurons) of a neural network (phenotype of a genome).

    Args:
        cid (int): The innovation number of the connection. As described in the
            original NEAT paper :cite:`stanley:ec02`, this serves as a
            historical marker for the gene, helping to identify homologous
            genes.
        from_node (NodeGene): Node from where the connection is originated. The
            source node of the connection.
        to_node (NodeGene): Node to where the connection is headed. The
            destination node of the connection.
        weight (float): The weight of the connection.
        enabled (bool): Whether the initial state of the newly created
            connection should enabled or disabled.

    Attributes:
        weight (float): The weight of the connection.
        enabled (bool): Whether the connection is enabled or not. A disabled
            connection won't be considered during the computations of the neural
            network.
    """

    def __init__(self,
                 cid: int,
                 from_node: NodeGene,
                 to_node: NodeGene,
                 weight: float,
                 enabled: bool = True) -> None:
        self._id = cid
        self._from_node = from_node
        self._to_node = to_node
        self.weight = weight
        self.enabled = enabled

    @property
    def id(self) -> int:
        """ Innovation number of the connection gene.

         As described in the original NEAT paper :cite:`stanley:ec02`, this
         value serves as a historical marker for the gene, helping to identify
         homologous genes. Although must of the identification is based on the
         nodes that form the connection, this ID is helpful to increase the
         speed of certain comparisons.
        """
        return self._id

    @property
    def from_node(self) -> NodeGene:
        """ Node where the connection is originated (source node). """
        return self._from_node

    @property
    def to_node(self) -> NodeGene:
        """ Node to where the connection is headed (destination node). """
        return self._to_node

    def self_connecting(self) -> bool:
        """
        Returns `True` if the connection is connecting a node to itself and
        `False` otherwise.
        """
        return self._from_node == self._to_node


def align_connections(
        con_list1: List[ConnectionGene],
        con_list2: List[ConnectionGene],
        print_alignment: bool = False
) -> Tuple[List[Optional[ConnectionGene]], List[Optional[ConnectionGene]]]:
    """ Aligns the matching connection genes of the given lists.

    In the context of NEAT :cite:`stanley:ec02`, aligning homologous connections
    genes is required both to compare the similarity of a pair of genomes and to
    perform sexual reproduction. Two connection genes are said to match or to be
    homologous if they have the same innovation ID, meaning that they represent
    the same structure.

    Genes that do not match are either disjoint or excess, depending on whether
    they occur within or outside the range of the other parent’s innovation
    numbers. They represent a structure that is not present in the other genome.

    Args:
        con_list1 (List[ConnectionGene]): The first list of connection genes.
        con_list2 (List[ConnectionGene]): The second list of connection genes.
        print_alignment: Whether to print the generated alignment or not. Used
            for debugging.

    Returns:
        A tuple containing two lists of the same size. Index 0 corresponds to
        the first list and index 1 to the second list. The returned lists
        contain connection genes or `None`. The order of the genes is preserved
        in the returned lists (but not their indices!).

        If, given a position, there are two genes (one in each list), the genes
        match. On the other hand, if, in the position, there is only one gene
        (on one of the lists) and a `None` value (on the other list), the genes
        are either disjoint or excess.
    """
    con_dict1 = {c.id: c for c in con_list1}
    con_dict2 = {c.id: c for c in con_list2}
    union = sorted(set(con_dict1.keys()) | set(con_dict2.keys()))

    aligned1, aligned2 = [], []
    for cid in union:
        aligned1.append(con_dict1[cid] if cid in con_dict1 else None)
        aligned2.append(con_dict2[cid] if cid in con_dict2 else None)

    # debug
    if print_alignment:
        for c1, c2 in zip(aligned1, aligned2):
            print(c1.id if c1 is not None else "-", end=" | ")
            print(c2.id if c2 is not None else "-")

    return aligned1, aligned2


class NodeIdException(Exception):
    """ Indicates that an attempt has been made to assign a new ID to a gene
    node that already has an ID.
    """
    pass


class ConnectionIdException(Exception):
    """ Indicates that an attempt has been made to assign a new ID to a
    connection gene that already has an ID.
    """
    pass


class NodeParentsException(Exception):
    """ Indicates that an attempt has been made to get the parents of a
    non-hidden node.
    """
    pass
