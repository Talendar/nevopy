"""
todo
"""

from neat.genes import *
import neat.activations as activations

import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout


class Genome:
    """ Linear representation of a neural network's connectivity.

    Attributes:
        todo
    """

    def __init__(self,
                 num_inputs,
                 num_outputs,
                 id_handler,
                 out_activation=activations.steepened_sigmoid,
                 hidden_activation=activations.steepened_sigmoid,
                 bias=1):
        """
        todo

        :param num_inputs:
        :param num_outputs:
        :param id_handler:
        :param out_activation:
        :param hidden_activation:
        :param bias:
        """
        self._id_handler = id_handler

        self._input_nodes = []
        self._hidden_nodes = []
        self._output_nodes = []
        self._bias_node = None

        self._connections = []
        self._hidden_activation = hidden_activation
        self._activated_nodes = None

        # init input nodes
        node_counter = 0
        for _ in range(num_inputs):
            self._input_nodes.append(NodeGene(node_id=node_counter,
                                              node_type=NodeGene.Type.INPUT))
            node_counter += 1

        # init bias node
        if bias is not None:
            self._bias_node = NodeGene(node_id=node_counter,
                                       node_type=NodeGene.Type.BIAS,
                                       initial_activation=bias)
            node_counter += 1

        # init output nodes
        for _ in range(num_outputs):
            out_node = NodeGene(node_id=node_counter,
                                node_type=NodeGene.Type.OUTPUT,
                                activation_func=out_activation)
            self._output_nodes.append(out_node)
            node_counter += 1

            # connecting all input nodes to all output nodes
            for in_node in self._input_nodes:
                self.add_connection(in_node, out_node)

    def add_connection(self, src_node, dest_node, weight=None):
        """ Adds a new connection between the two given nodes.

        :param src_node: source node (where the connection is coming from).
        :param dest_node: destination node (where the connection is headed to)
        :param weight: weight of the connection; if None, a random weight between -1 and 1 is chosen.
        :return: False if the connection couldn't be added (because it already exists) or True if the connection was
        successfully added.
        """
        if connection_exists(src_node, dest_node):
            return False

        connection = ConnectionGene(inov_id=self._id_handler.connection_id(src_node.id, dest_node.id),
                                    from_node=src_node, to_node=dest_node,
                                    weight=np.random.uniform(-1, 1) if weight is None else weight)  # todo: custom interval (maybe add a parameter?)
        self._connections.append(connection)
        dest_node.in_connections.append(connection)
        return True

    def add_hidden_node(self):
        """ Adds a new hidden node to the genome.

        This method implements the "add node mutation" procedure described in the original NEAT paper.

        "An existing connection is split and the new node placed where the old connection used to be. The old connection
        is disabled and two new connections are added to the genome. The new connection leading into the new node
        receives a weight of 1, and the new connection leading out receives the same weight as the old connection."
        - Stanley, K. O. & Miikkulainen, R. (2002)
        """
        original_connection = np.random.choice([c for c in self._connections if c.enabled])
        original_connection.enabled = False

        src_node, dest_node = original_connection.from_node, original_connection.to_node
        new_node = NodeGene(node_id=self._id_handler.hidden_node_id(src_node.id, dest_node.id),
                            node_type=NodeGene.Type.HIDDEN,
                            activation_func=self._hidden_activation)
        self._hidden_nodes.append(new_node)
        self.add_connection(src_node, new_node, weight=1)
        self.add_connection(new_node, dest_node, weight=original_connection.weight)

    def _process_node(self, n):
        """ Recursively processes the activation of the given node.

        :param n: the instance of NodeGene to be processed.
        :return: the current value of the activation of n.
        """
        if (n.type != NodeGene.Type.INPUT
                and n.type != NodeGene.Type.BIAS
                and not self._activated_nodes[n.id]):  # checks if the node needs to be activated
            # activating the node
            # the current node (n) is immediately marked as activated; this is needed due to recurrency: if, during the
            # recursive calls, some node m depends on the activation of n, the old activation of n is used
            self._activated_nodes[n.id] = True
            zsum = 0
            for connection in n.in_connections:
                if connection.enabled:
                    in_node, weight = connection.from_node, connection.weight
                    zsum += weight * self._process_node(in_node)
            n.activate(zsum)

        return n.activation

    def process(self, X):
        """ Processes the given input using the neural network (phenotype) encoded in the genome.

        :param X: input to be fed to the neural network.
        :return: numpy array with the activations of the output nodes/neurons.
        """
        # preparing input nodes
        for n, x in zip(self._input_nodes, X):
            n.activate(x)

        # resetting activated nodes dict
        self._activated_nodes = {n.id: False for n in self._output_nodes + self._hidden_nodes}

        # processing nodes in a top-down manner (starts from the output nodes)
        # note that nodes not connected to at least one output node are not processed
        h = np.zeros(len(self._output_nodes))
        for i, out_node in enumerate(self._output_nodes):
            h[i] = self._process_node(out_node)

        return h

    def visualize(self,
                  figsize=(12, 8),
                  show_node_id=True,
                  input_color="deepskyblue",
                  output_color="mediumseagreen",
                  hidden_color="silver",
                  bias_color="khaki"):
        """
        Plots the neural network (phenotype) encoded by the genome.

        :param figsize: size of the matplotlib figure.
        :param show_node_id: if True, the nodes will have their id drawn inside them.
        :param input_color: color for the input nodes.
        :param output_color: color for the output nodes.
        :param hidden_color: color for the hidden nodes.
        :param bias_color: color for the bias node.
        :return:
        """
        graph = nx.DiGraph()
        plt.figure(figsize=figsize)
        nodes_colors = []

        # input nodes
        for i, in_node in enumerate(self._input_nodes):
            graph.add_node(in_node.id)
            nodes_colors.append(input_color)

        # bias node
        if self._bias_node is not None:
            graph.add_node(self._bias_node.id)
            nodes_colors.append(bias_color)

        # output nodes
        for i, out_node in enumerate(self._output_nodes):
            graph.add_node(out_node.id)
            nodes_colors.append(output_color)

        # connections and hidden nodes
        for c in self._connections:
            graph.add_edge(c.from_node.id, c.to_node.id, weight=c.weight)

        # calculating edge colors
        edges_weights = list(nx.get_edge_attributes(graph, "weight").values())
        min_w, max_w = np.min(edges_weights), np.max(edges_weights)
        edges_colors = [(1, 0.6*(1 - (w - min_w) / (max_w - min_w)), 0, 0.3 + 0.7*(w - min_w) / (max_w - min_w))
                        for w in edges_weights]

        # plotting
        pos = graphviz_layout(graph, prog='dot', args="-Grankdir=LR")
        nodes_colors += [hidden_color] * len(self._hidden_nodes)

        nx.draw(graph, with_labels=show_node_id, pos=pos, node_color=nodes_colors, edge_color=edges_colors)
        plt.show()
