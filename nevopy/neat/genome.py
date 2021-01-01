"""
todo
"""

import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

import nevopy.activations as activations
from nevopy.neat.genes import *
from nevopy.math import chance

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
        self.fitness = 0

        self._input_nodes = []
        self._hidden_nodes = []
        self._output_nodes = []
        self._bias_node = None

        self._connections = []
        self._output_activation = out_activation
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

    @property
    def args(self):
        """ Returns the arguments to be used when instantiating a new genome based on this one."""
        return {"num_inputs": len(self._input_nodes),
                "num_outputs": len(self._output_nodes),
                "id_handler": self._id_handler,
                "out_activation": self._output_activation,
                "hidden_activation": self._hidden_activation,
                "bias": self._bias_node.activation}

    def add_connection(self, src_node, dest_node, weight="random", rdm_weight_interval=(-1, 1)):
        """ Adds a new connection between the two given nodes.

        :param src_node: source node (where the connection is coming from).
        :param dest_node: destination node (where the connection is headed to)
        :param weight: weight of the connection; if "random", a random weight within the given interval is chosen.
        :param rdm_weight_interval: interval that contains the random chosen weight; this parameter is ignored if you
        pass a pre-defined weight as argument.
        :raise ConnectionExistsError: if the connection already exists.
        :raise ConnectionToBiasNodeError: if the destination node is a bias node.
        """
        if connection_exists(src_node, dest_node):
            raise ConnectionExistsError(f"Attempt to create an already existing connection "
                                        f"({src_node.id}->{dest_node.id}).")
        if dest_node.type == NodeGene.Type.BIAS:
            raise ConnectionToBiasNodeError(f"Attempt to create a connection pointing to a bias node "
                                            f"({src_node.id}->{dest_node.id}). Nodes of this type don't process input.")
        connection = ConnectionGene(inov_id=self._id_handler.connection_id(src_node.id, dest_node.id),
                                    from_node=src_node, to_node=dest_node,
                                    weight=np.random.uniform(*rdm_weight_interval) if weight == "random" else weight)
        self._connections.append(connection)
        src_node.out_connections.append(connection)
        dest_node.in_connections.append(connection)

    def add_random_connection(self, allow_self_connections=True):
        """ Adds a new connection between two nodes in the genome.

        :param allow_self_connections: if True, a node is allowed to connect to itself (recurrent connection).
        :return: a tuple containing the source node and the destination node of the connection if a new connection was
        successfully created; None if there was no space for a new connection.
        """
        all_src_nodes = self.nodes()
        np.random.shuffle(all_src_nodes)

        all_dest_nodes = [n for n in all_src_nodes if n.type != NodeGene.Type.BIAS]  # removing bias nodes from dest
        np.random.shuffle(all_dest_nodes)

        for src_node in all_src_nodes:
            for dest_node in all_dest_nodes:
                if src_node != dest_node or allow_self_connections:
                    try:
                        self.add_connection(src_node, dest_node)
                        return src_node, dest_node
                    except ConnectionExistsError:
                        pass
        return None

    def add_random_hidden_node(self):
        """ Adds a new hidden node to the genome in a random position.

        This method implements the "add node mutation" procedure described in the original NEAT paper.

        "An existing connection is split and the new node placed where the old connection used to be. The old connection
        is disabled and two new connections are added to the genome. The new connection leading into the new node
        receives a weight of 1, and the new connection leading out receives the same weight as the old connection."
        - Stanley, K. O. & Miikkulainen, R. (2002)

        :return: the newly created node.
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
        return new_node

    def mutate_weights(self,
                       mutation_chance=0.8,
                       perturbation_pc=0.1,
                       reset_chance=0.1,
                       reset_weight_interval=(-1, 1)):
        """
        Randomly mutates the weights of the encoded network's connections.

        :param mutation_chance: chance of mutating a connection.
        :param perturbation_pc: defines the maximum absolute percentage value for the perturbation of the weights.
        :param reset_chance: chance to reset the weight of a connection (assign it a new random value).
        :param reset_weight_interval: interval from which the reset weight new value will be drawn from.
        """
        for connection in self._connections:
            if chance(mutation_chance):  # checking whether the weight will be mutated
                if chance(reset_chance):  # checking whether the weight will be reset
                    connection.weight = np.random.uniform(*reset_weight_interval)
                else:
                    d = connection.weight * np.random.uniform(low=-perturbation_pc, high=perturbation_pc)
                    connection.weight += d

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
                    src_node, weight = connection.from_node, connection.weight
                    zsum += weight * self._process_node(src_node)
            n.activate(zsum)
        return n.activation

    def process(self, X):
        """ Processes the given input using the neural network (phenotype) encoded in the genome.

        :param X: input to be fed to the neural network.
        :return: numpy array with the activations of the output nodes/neurons.
        """
        # preparing input nodes
        assert len(X) == len(self._input_nodes), "The input size must match the number of input nodes in the network!"
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

    def nodes(self):
        """ Returns all the genome's nodes genes. Order: inputs, bias, outputs and hiddens. """
        return self._input_nodes + \
            ([self._bias_node] if self._bias_node is not None else []) + \
            self._output_nodes + \
            self._hidden_nodes

    def info(self):
        txt = ">> NODES ACTIVATIONS\n"
        for n in self.nodes():
            txt += f"[{n.id}][{str(n.type).split('.')[1][0]}] {n.activation}\n"
        txt += "\n>> CONNECTIONS\n"
        for c in self._connections:
            txt += f"[{'ON' if c.enabled else 'OFF'}][{c.from_node.id}->{c.to_node.id}] {c.weight}\n"
        return txt

    def visualize(self,
                  show=True,
                  save_to=None,
                  save_transparent=False,
                  figsize=(10, 6),
                  pad=1,
                  legends=True,
                  nodes_ids=True,
                  node_id_color="black",
                  background_color="snow",
                  legend_box_color="honeydew",
                  input_color="deepskyblue",
                  output_color="mediumseagreen",
                  hidden_color="silver",
                  bias_color="khaki"):
        """
        Plots the neural network (phenotype) encoded by the genome.

        For the colors parameters, it's possible to pass a string with the color HEX value or a string with the color's
        name (names available here: https://matplotlib.org/3.1.0/gallery/color/named_colors.html).

        :param show: whether to show the image or not.
        :param save_to: path to save the image.
        :param save_transparent: if True, the saved image will have a transparent background.
        :param figsize: size of the matplotlib figure.
        :param pad: the image's padding (distance between the figure and the image's border).
        :param legends: if True, a box with legends describing the nodes colors will be drawn.
        :param nodes_ids: if True, the nodes will have their ID drawn inside them.
        :param node_id_color: color for nodes id.
        :param background_color: color for the plot's background.
        :param legend_box_color: color for the legends box.
        :param input_color: color for the input nodes.
        :param output_color: color for the output nodes.
        :param hidden_color: color for the hidden nodes.
        :param bias_color: color for the bias node.
        :return:
        """
        assert show or save_to is not None
        plt.rcParams['axes.facecolor'] = background_color

        G = nx.MultiDiGraph()
        G.add_nodes_from([n.id for n in self.nodes()])
        plt.figure(figsize=figsize)

        # connections
        for c in self._connections:
            if c.enabled:
                G.add_edge(c.from_node.id, c.to_node.id, weight=c.weight)

        # calculating edge colors
        edges_weights = list(nx.get_edge_attributes(G, "weight").values())
        min_w, max_w = np.min(edges_weights), np.max(edges_weights)
        edges_colors = [(1, 0.6*(1 - (w - min_w) / (max_w - min_w)), 0, 0.3 + 0.7*(w - min_w) / (max_w - min_w))
                        for w in edges_weights]

        # plotting
        pos = graphviz_layout(G, prog='dot', args="-Grankdir=LR")
        nx.draw_networkx_nodes(G, pos=pos, nodelist=[n.id for n in self._input_nodes],
                               node_color=input_color, label='Input nodes')
        nx.draw_networkx_nodes(G, pos=pos, nodelist=[n.id for n in self._output_nodes],
                               node_color=output_color, label='Output nodes')
        nx.draw_networkx_nodes(G, pos=pos, nodelist=[n.id for n in self._hidden_nodes],
                               node_color=hidden_color, label='Hidden nodes')
        if self._bias_node is not None:
            nx.draw_networkx_nodes(G, pos=pos, nodelist=[self._bias_node.id],
                                   node_color=bias_color, label='Bias node')

        nx.draw_networkx_edges(G, pos=pos, edge_color=edges_colors)
        if nodes_ids:
            nx.draw_networkx_labels(G, pos, labels={k.id: k.id for k in self.nodes()}, font_size=10,
                                    font_color=node_id_color, font_family='sans-serif')
        if legends:
            plt.legend(facecolor=legend_box_color, borderpad=0.8, labelspacing=0.5)

        plt.tight_layout(pad=pad)
        if save_to is not None:
            plt.savefig(save_to, transparent=save_transparent)

        if show:
            plt.show()

    # noinspection PyProtectedMember
    @staticmethod
    def mate_genomes(gen1, gen2):
        """ todo

        :param gen1:
        :param gen2:
        :return:
        """
        # make gen1 point to the genome with the highest fitness
        if gen1.fitness < gen2.fitness:
            gen1, gen2 = gen2, gen1

        # sorting connections by their id
        prim_con = sorted(gen1._connections, key=lambda c: c.id)
        sec_con = sorted(gen2._connections, key=lambda c: c.id)

        # new genome
        new_gen = Genome(**gen1.args)

        # mate
        i = 0
        for c1 in prim_con:
            while sec_con[i].id < c1.id:
                i += 1
            c2 = sec_con[i]
            chosen_c = c1 if c1.id != c2.id else np.random.choice((c1, c2))
            # todo: add nodes (if needed) and add the new connection between them


class ConnectionExistsError(Exception):
    """ Exception which indicates that a connection between two given nodes already exists. """
    pass


class ConnectionToBiasNodeError(Exception):
    """
    Exception which indicates that an attempt has been made to create a connection containing a bias node as  destination.
    """
    pass
