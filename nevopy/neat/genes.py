"""
todo
"""


from enum import Enum
import uuid


class NodeGene:
    """ Represents a single neuron (node) in a neural network.

    Attributes:
        todo
    """

    def __init__(self,
                 node_id,
                 node_type,
                 activation_func,
                 initial_activation,
                 parent_connection_nodes=None,
                 debug_info=None):
        """
        todo

        :param node_id:
        :param node_type:
        :param activation_func:
        :param initial_activation:
        """
        self._id = node_id
        self._type = node_type
        self._initial_activation = initial_activation
        self._activation = initial_activation
        self._function = activation_func
        self._parent_connection_nodes = parent_connection_nodes
        self.in_connections = []
        self.out_connections = []
        self._temp_id = None if self._id is not None else uuid.uuid4().hex[:10]
        self.debug_info = debug_info

    class Type(Enum):
        """ Specifies the possible types of node genes. """
        INPUT, BIAS, HIDDEN, OUTPUT = range(4)

    @property
    def id(self):
        return self._id if self._id is not None else self._temp_id

    def is_id_temp(self):
        return self._id is None

    @id.setter
    def id(self, new_id):
        if self._id is not None:
            raise NodeIdException("Attempt to assign a new ID to a node gene that already has an ID!")
        self._id = new_id
        self._temp_id = None

    @property
    def type(self):
        return self._type

    @property
    def activation(self):
        return self._activation

    @property
    def parent_connection_nodes(self):
        if self._type != NodeGene.Type.HIDDEN:
            raise NodeParentsException("Attempt to get the parents of a non-hidden node!")
        return self._parent_connection_nodes

    def activate(self, x):
        """ Applies the node's activation function to the given input, updating the node's activation (output). """
        self._activation = self._function(x)

    def shallow_copy(self, debug_info=None):
        """ Creates a new node equal to this one except for the connections. """
        return NodeGene(node_id=self._id,
                        node_type=self._type,
                        activation_func=self._function,
                        initial_activation=self._initial_activation,
                        parent_connection_nodes=self._parent_connection_nodes,
                        debug_info=debug_info)

    def reset_activation(self):
        """ Resets the node's activation to its initial value. """
        self._activation = self._initial_activation


class ConnectionGene:
    """ Represents the connection/link between two node genes.

    Attributes:
        todo
    """

    def __init__(self, cid, from_node, to_node, weight, enabled=True, debug_info=None):
        """ todo

        :param cid:
        :param from_node:
        :param to_node:
        :param weight:
        """
        self._id = cid
        self._from_node = from_node
        self._to_node = to_node
        self.weight = weight
        self.enabled = enabled
        self.debug_info = debug_info

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, new_id):
        if self._id is not None:
            raise ConnectionIdException("Attempt to assign a new ID to a connection gene that already has an ID!")
        self._id = new_id

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
    for dest_cin, src_cout in zip(dest_node.in_connections, src_node.out_connections):
        if dest_cin.from_node.id == src_node.id or src_cout.to_node.id == dest_node.id:
            return True
    return False


def align_connections(con_list1, con_list2, print_alignment=False):
    """ Aligns the connection genes on both lists. Disjunctions and excesses are filled with None. """
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
    """ Indicates that an attempt has been made to assign a new ID to a gene node that already has an ID. """
    pass


class ConnectionIdException(Exception):
    """ Indicates that an attempt has been made to assign a new ID to a connection gene that already has an ID. """
    pass


class NodeParentsException(Exception):
    pass
