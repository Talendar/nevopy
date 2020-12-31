"""
todo
"""


from enum import Enum
from nevopy.activations import linear


class NodeGene:
    """ Represents a single neuron (node) in a neural network.

    Attributes:
        todo
    """

    def __init__(self,
                 node_id,
                 node_type,
                 activation_func=linear,
                 initial_activation=0):
        """
        todo

        :param node_id:
        :param node_type:
        :param activation_func:
        :param initial_activation:
        """
        self._id = node_id
        self._type = node_type
        self._activation = initial_activation
        self._function = activation_func
        self.in_connections = []
        self.out_connections = []

    class Type(Enum):
        """ Specifies the possible types of node genes. """
        INPUT, BIAS, HIDDEN, OUTPUT = range(4)

    @property
    def id(self):
        return self._id

    @property
    def type(self):
        return self._type

    @property
    def activation(self):
        return self._activation

    def activate(self, x):
        """ Applies the node's activation function to the given input, updating the node's activation (output). """
        self._activation = self._function(x)


class ConnectionGene:
    """ Represents the connection/link between two node genes.

    Attributes:
        todo
    """

    def __init__(self, inov_id, from_node, to_node, weight):
        """ todo

        :param inov_id:
        :param from_node:
        :param to_node:
        :param weight:
        """
        self._id = inov_id
        self._from_node = from_node
        self._to_node = to_node
        self.weight = weight
        self.enabled = True

    @property
    def id(self):
        return self._id

    @property
    def from_node(self):
        return self._from_node

    @property
    def to_node(self):
        return self._to_node


def connection_exists(src_node, dest_node):
    """ todo

    :param src_node:
    :param dest_node:
    :return:
    """
    for connection in dest_node.in_connections:
        if connection.from_node.id == src_node.id:
            return True
    return False
