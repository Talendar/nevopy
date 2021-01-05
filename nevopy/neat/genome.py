"""
todo
"""

import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

import nevopy.activations as activations
from nevopy.neat.genes import *
import nevopy.utils as utils

import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from timeit import default_timer as timer


class Genome:
    """ Linear representation of a neural network's connectivity.

    todo
    """

    start_time = timer()  # todo: delete

    def __init__(self,
                 num_inputs,
                 num_outputs,
                 config,
                 genome_id=None,
                 initial_connections=True):
        """
        todo

        :param num_inputs:
        :param num_outputs:
        """
        self.config = config
        self._id = genome_id
        self.species_id = None

        self.new_connections = []
        self.new_nodes = []
        self.new_genomes = []
        self._hid_connections_cache = {}
        self._activated_nodes = None

        self.fitness = 0
        self.adj_fitness = 0

        self._input_nodes = []
        self._hidden_nodes = []
        self._output_nodes = []
        self._bias_node = None

        self.connections = []
        self._output_activation = self.config.out_nodes_activation
        self._hidden_activation = self.config.hidden_nodes_activation

        # init input nodes
        node_counter = 0
        for _ in range(num_inputs):
            self._input_nodes.append(NodeGene(node_id=node_counter,
                                              node_type=NodeGene.Type.INPUT,
                                              activation_func=activations.linear,
                                              initial_activation=self.config.initial_node_activation,
                                              debug_info="__init__ genome"))
            node_counter += 1

        # init bias node
        if self.config.bias_value is not None:
            self._bias_node = NodeGene(node_id=node_counter,
                                       node_type=NodeGene.Type.BIAS,
                                       activation_func=activations.linear,
                                       initial_activation=self.config.bias_value,
                                       debug_info="__init__ genome")
            node_counter += 1

        # init output nodes
        connection_counter = 0
        for _ in range(num_outputs):
            out_node = NodeGene(node_id=node_counter,
                                node_type=NodeGene.Type.OUTPUT,
                                activation_func=self._output_activation,
                                initial_activation=self.config.initial_node_activation,
                                debug_info="__init__ genome")
            self._output_nodes.append(out_node)
            node_counter += 1

            # connecting all input nodes to all output nodes
            if initial_connections:
                for in_node in self._input_nodes:
                    connection_counter += 1
                    self.add_connection(in_node, out_node, cid=connection_counter, debug_info="__init__ genome")

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, new_id):
        if self._id is not None:
            raise GenomeIdException("Attempt to assign a new ID to a genome that already has an ID!")
        self._id = new_id

    def reset_activations(self):
        """ Resets the genome's nodes cached activations."""
        self._activated_nodes = None
        for n in self.nodes():
            n.reset_activation()

    def reset_news_cache(self):
        """
        Resets the genome's cache with newly created nodes, connections and copies. This should be called at the
        beginning of each new generation.
        """
        self.new_nodes = []
        self.new_connections = []
        self.new_genomes = []

    def reset_connections_ids_cache(self):
        """
        Resets the genome's internal innovations IDs cache. This should be called whenever the id handler is reset.
        """
        self._hid_connections_cache = {}

    def distance(self, other):
        """
        Calculates and returns the distance between two genomes (the lower the distance, the higher the compatibility
        between the genomes).

        The formula used is the one presented in the original NEAT paper:
            distance = (c1 * E / N) + (c2 * D / N) + c3 * W
        """
        genes = align_connections(self.connections, other.connections)
        excess = disjoint = weight_diff = num_matches = 0

        g1_max_innov = np.amax([c.id for c in self.connections])
        g2_max_innov = np.amax([c.id for c in other.connections])

        for c1, c2 in zip(*genes):
            # non-matching genes:
            if c1 is None or c2 is None:
                if (c1 is None and c2.id > g1_max_innov) or (c2 is None and c1.id > g2_max_innov):
                    excess += 1
                else:
                    disjoint += 1
            # matching genes:
            else:
                num_matches += 1
                weight_diff += abs(c1.weight - c2.weight)

        c1, c2 = self.config.excess_genes_coefficient, self.config.disjoint_genes_coefficient
        c3 = self.config.weight_difference_coefficient
        n = max(len(self.connections), len(other.connections))
        return ((c1 * excess + c2 * disjoint) / n) + c3 * weight_diff / num_matches

    def add_connection(self, src_node, dest_node, enabled=True, cid=None, weight=None, debug_info=None):
        """ Adds a new connection between the two given nodes.

        :param src_node: source node (where the connection is coming from).
        :param dest_node: destination node (where the connection is headed to)
        :param enabled: whether the connection should start enabled or not.
        :param cid: the id (int)  of the new connection; if None, a new id will be chosen by the id handler.
        :param weight: weight of the connection; if None, a random weight within the given interval is chosen.
        :param debug_info: -
        :raise ConnectionExistsError: if the connection already exists.
        :raise ConnectionToBiasNodeError: if the destination node is a bias node.
        """
        if connection_exists(src_node, dest_node):
            raise ConnectionExistsError(f"Attempt to create an already existing connection "
                                        f"({src_node.id}->{dest_node.id}).")
        if dest_node.type == NodeGene.Type.BIAS or dest_node.type == NodeGene.Type.INPUT:
            raise ConnectionToBiasNodeError(f"Attempt to create a connection pointing to a bias or input node "
                                            f"({src_node.id}->{dest_node.id}). Nodes of this type don't process input.")

        connection = ConnectionGene(
            cid=cid,  # TODO: check id handler removal
            from_node=src_node, to_node=dest_node,
            weight=np.random.uniform(*self.config.new_weight_interval) if weight is None else weight,
            debug_info=debug_info)

        if cid is None:
            self.new_connections.append(connection)

        connection.enabled = enabled
        self.connections.append(connection)
        src_node.out_connections.append(connection)
        dest_node.in_connections.append(connection)

    def add_random_connection(self):
        """ Adds a new connection between two nodes in the genome.

        :return: a tuple containing the source node and the destination node of the connection if a new connection was
        successfully created; None if there was no space for a new connection.
        """
        all_src_nodes = self.nodes()
        np.random.shuffle(all_src_nodes)

        all_dest_nodes = [n for n in all_src_nodes if n.type != NodeGene.Type.BIAS and n.type != NodeGene.Type.INPUT]
        np.random.shuffle(all_dest_nodes)

        for src_node in all_src_nodes:
            for dest_node in all_dest_nodes:
                if src_node != dest_node or self.config.allow_self_connections:
                    try:
                        self.add_connection(src_node, dest_node, debug_info="add_random_connection")
                        return src_node, dest_node
                    except ConnectionExistsError:
                        pass
        return None

    def enable_random_connection(self):
        """ Randomly enable a disabled connection gene. """
        disabled = [c for c in self.connections if not c.enabled]
        if len(disabled) > 0:
            connection = np.random.choice(disabled)
            connection.enabled = True

    def _eligible_hidden_node_parents(self, src, dest):
        try:
            return dest.id not in self._hid_connections_cache[src.id]
        except KeyError:
            return True

    def add_random_hidden_node(self, cached_hids=None):
        """ Adds a new hidden node to the genome in a random position.

        This method implements the "add node mutation" procedure described in the original NEAT paper:

        "An existing connection is split and the new node placed where the old connection used to be. The old connection
        is disabled and two new connections are added to the genome. The new connection leading into the new node
        receives a weight of 1, and the new connection leading out receives the same weight as the old connection."
        - Stanley, K. O. & Miikkulainen, R. (2002)

        :return: the newly created node.
        """
        eligible_connections = [c for c in self.connections
                                if self._eligible_hidden_node_parents(c.from_node, c.to_node)]
        if not eligible_connections:
            return None

        original_connection = np.random.choice(eligible_connections)
        original_connection.enabled = False
        src_node, dest_node = original_connection.from_node, original_connection.to_node

        # checking cached ids
        new_node_id = None
        if cached_hids is not None:
            try:
                new_node_id = cached_hids[src_node.id][dest_node.id]
            except KeyError:
                pass

        # creating new hidden node
        new_node = NodeGene(node_id=new_node_id,
                            node_type=NodeGene.Type.HIDDEN,
                            activation_func=self._hidden_activation,
                            initial_activation=self.config.initial_node_activation,
                            parent_connection_nodes=(src_node, dest_node),
                            debug_info="add_random_hidden_node")

        self._hidden_nodes.append(new_node)
        self.new_nodes.append(new_node)

        # adding connections
        self.add_connection(src_node, new_node, weight=1,
                            debug_info=f"[add_random_hidden_node | {(timer() - self.start_time) * 1000}] "
                                       f"src_id={src_node.id} , new_id={new_node.id}, dest_id={dest_node.id}"
                                       f"   (hid_connections_cache={self._hid_connections_cache}")
        self.add_connection(new_node, dest_node, weight=original_connection.weight,
                            debug_info=f"[add_random_hidden_node | {(timer() - self.start_time) * 1000}] "
                                       f"src_id={src_node.id} , new_id={new_node.id}, dest_id={dest_node.id}"
                                       f"   < hid_connections_cache={self._hid_connections_cache} >")

        # caching innovation
        if src_node.id not in self._hid_connections_cache:
            self._hid_connections_cache[src_node.id] = set()

        self._hid_connections_cache[src_node.id].add(dest_node.id)
        return new_node

    def mutate_weights(self):
        """ Randomly mutates the weights of the encoded network's connections. """
        for connection in self.connections:
            if utils.chance(self.config.weight_reset_chance):  # checking whether the weight will be reset
                connection.weight = np.random.uniform(*self.config.new_weight_interval)
            else:
                d = connection.weight * np.random.uniform(
                    low=-self.config.weight_perturbation_pc, high=self.config.weight_perturbation_pc)
                connection.weight += d

    def delete_duplicated_connections(self):
        pass

    def shallow_copy(self):
        """ Returns a copy of this genome without the hidden nodes and connections (the copy has no ID). """
        new_genome = Genome(num_inputs=len(self._input_nodes), num_outputs=len(self._output_nodes),
                            config=self.config, initial_connections=False, genome_id=None)
        new_genome._hid_connections_cache = dict(self._hid_connections_cache)
        self.new_genomes.append(new_genome)
        return new_genome

    def deep_copy(self):
        """ Returns an exact copy of this genome (except for the ID; the copy doesn't have an ID). """
        new_genome = self.shallow_copy()
        copied_nodes = {n.id: n for n in new_genome.nodes()}

        # creating required nodes
        for node in self._hidden_nodes:
            new_node = node.shallow_copy()
            copied_nodes[node.id] = new_node
            new_genome._hidden_nodes.append(new_node)

        # adding connections
        for c in self.connections:
            try:
                new_genome.add_connection(src_node=copied_nodes[c.from_node.id],
                                          dest_node=copied_nodes[c.to_node.id],
                                          enabled=c.enabled, cid=c.id, weight=c.weight,
                                          debug_info=f"[deep_copy | {(timer() - self.start_time) * 1000}] "
                                                     f"src_id={c.from_node.id} , dest_id={c.to_node.id}")
            except ConnectionExistsError:
                cons = [f"[{con.id}] {con.from_node.id}->{con.to_node.id} ({con.enabled})" for con in self.connections]
                db_info = "\t" + "\n\t".join([con.debug_info for con in self.connections if con.id == c.id])
                raise ConnectionExistsError(
                    "Connection exists error when duplicating parent.\n"
                    f"c.debug info: {c.debug_info}\n"
                    f"duplicates debug info: \n{db_info}\n"
                    f"Source node's connections: {cons}")

        return new_genome

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
        """ Returns all the genome's nodes genes. Order: inputs, bias, outputs and hidden. """
        return self._input_nodes + \
            ([self._bias_node] if self._bias_node is not None else []) + \
            self._output_nodes + \
            self._hidden_nodes

    def info(self):
        txt = ">> NODES ACTIVATIONS\n"
        for n in self.nodes():
            txt += f"[{n.id}][{str(n.type).split('.')[1][0]}] {n.activation}\n"
        txt += "\n>> CONNECTIONS\n"
        for c in self.connections:
            txt += f"[{'ON' if c.enabled else 'OFF'}][{c.id}][{c.from_node.id}->{c.to_node.id}] {c.weight}\n"
        return txt

    def visualize(self,
                  show=True,
                  block_thread=True,
                  save_to=None,
                  save_transparent=False,
                  figsize=(10, 6),
                  pad=1,
                  legends=True,
                  nodes_ids=True,
                  node_id_color="black",
                  edge_curviness=0.1,
                  edges_ids=False,
                  edge_id_color="black",
                  background_color="snow",
                  legend_box_color="honeydew",
                  input_color="deepskyblue",
                  output_color="mediumseagreen",
                  hidden_color="silver",
                  bias_color="khaki"):
        """
        Plots the neural network (phenotype) encoded by the genome.

        Self-connecting edges are not drawn.

        For the colors parameters, it's possible to pass a string with the color HEX value or a string with the color's
        name (names available here: https://matplotlib.org/3.1.0/gallery/color/named_colors.html).

        :param show: whether to show the image or not.
        :param block_thread: whether to block the execution's thread while showing the image; useful for visualizing
        multiple genomes at once.
        :param save_to: path to save the image.
        :param save_transparent: if True, the saved image will have a transparent background.
        :param figsize: size of the matplotlib figure.
        :param pad: the image's padding (distance between the figure and the image's border).
        :param legends: if True, a box with legends describing the nodes colors will be drawn.
        :param nodes_ids: if True, the nodes will have their ID drawn inside them.
        :param node_id_color: color for nodes id.
        :param edge_curviness: angle, in radians, of the edges arcs; 0 is a straight line.
        :param edges_ids: if True, each connection will have its ID drawn on it; some labels might overlap with each
        other, making only one of them visible.
        :param edge_id_color: color of the connections ids.
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
        edges_labels = {}
        for c in self.connections:
            if c.enabled:
                G.add_edge(c.from_node.id, c.to_node.id, weight=c.weight)
                edges_labels[(c.from_node.id, c.to_node.id)] = c.id

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

        if G.number_of_edges() > 0:
            # calculating edges colors
            edges_weights = list(nx.get_edge_attributes(G, "weight").values())
            min_w, max_w = np.min(edges_weights), np.max(edges_weights)
            edges_colors = [(1, 0.6 * (1 - (w - min_w) / (max_w - min_w)), 0, 0.3 + 0.7 * (w - min_w) / (max_w - min_w))
                            for w in edges_weights]
            # drawing edges
            nx.draw_networkx_edges(G, pos=pos, edge_color=edges_colors,
                                   connectionstyle=f"arc3, rad={edge_curviness}")

        if edges_ids:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edges_labels, font_color=edge_id_color)

        if nodes_ids:
            nx.draw_networkx_labels(G, pos, labels={k.id: k.id for k in self.nodes()}, font_size=10,
                                    font_color=node_id_color, font_family='sans-serif')
        if legends:
            plt.legend(facecolor=legend_box_color, borderpad=0.8, labelspacing=0.5)

        plt.tight_layout(pad=pad)
        if save_to is not None:
            plt.savefig(save_to, transparent=save_transparent)

        if show:
            plt.show(block=block_thread)


# noinspection PyProtectedMember
def mate_genomes(gen1, gen2):
    """ Mates the two genomes to produce a new genome (offspring).

    Follows the idea described in the original paper of the NEAT algorithm:

    "When crossing over, the genes in both genomes with the same innovation numbers are lined up. These genes are called
    matching genes. (...). Matching genes are inherited randomly, whereas disjoint genes (those that do not match in the
    middle) and excess genes (those that do not match in the end) are inherited from the more fit parent. (...) [If the
    parents fitness are equal] the disjoint and excess genes are also inherited randomly. (...) there’s a preset chance
    that an inherited gene is disabled if it is disabled in either parent." - Stanley, K. O. & Miikkulainen, R. (2002)

    :param gen1: instance of Genome.
    :param gen2: instance of Genome.
    :return: the new genome (offspring).
    """
    # aligning matching genes
    genes = align_connections(gen1.connections, gen2.connections)

    # new genome
    new_gen = gen1.shallow_copy()
    new_gen._hid_connections_cache = {**gen2._hid_connections_cache, **gen1._hid_connections_cache}
    copied_nodes = {n.id: n for n in new_gen.nodes()}

    # mate (choose new genome's connections)
    chosen_connections = []
    for c1, c2 in zip(*genes):
        if c1 is None and gen1.adj_fitness > gen2.adj_fitness:
            # case 1: the gene is missing on gen1 and gen1 is dominant (higher fitness); action: ignore the gene
            continue

        if c2 is None and gen2.adj_fitness > gen1.adj_fitness:
            # case 2: the gene is missing on gen2 and gen2 is dominant (higher fitness); action: ignore the gene
            continue

        # case 3: the gene is missing either on gen1 or on gen2 and their fitness are equal; action: random choice
        # case 4: the gene is present both on gen1 and on gen2; action: random choice

        c = np.random.choice((c1, c2))
        if c is not None:
            # if the gene is disabled in either parent, it has a chance to also be disabled in the new genome
            enabled = (((c1 is not None and not c1.enabled) or (c2 is not None and not c2.enabled))
                       and utils.chance(gen1.config.disable_inherited_connection_chance))

            chosen_connections.append((c, enabled))

            # adding the hidden nodes of the connection (if needed)
            for node in (c.from_node, c.to_node):
                if node.type == NodeGene.Type.HIDDEN and node.id not in copied_nodes:
                    new_node = node.shallow_copy(debug_info="mate_genomes")
                    new_gen._hidden_nodes.append(new_node)
                    copied_nodes[node.id] = new_node

    # todo: solve new genomes being created without connections
    # this is not a bug, but it does add useless genomes to the population

    # adding inherited connections
    for c, enabled in chosen_connections:
        src_node, dest_node = copied_nodes[c.from_node.id], copied_nodes[c.to_node.id]
        try:
            new_gen.add_connection(src_node=src_node,
                                   dest_node=dest_node,
                                   enabled=enabled, cid=c.id, weight=c.weight,
                                   debug_info="mate_genomes")
        except ConnectionExistsError:
            # if this exception is raised, it means that the connection was already inherited from the other parent;
            # this is possible because, in some cases, a connection between the same two nodes appears in different
            # generations and are assigned, because of that, different IDs.
            pass
            # __debug_mating(genes, c, gen1, gen2, new_gen)
            # raise ConnectionExistsError()
    return new_gen


def __debug_mating(genes, c, gen1, gen2, new_gen):
    alignment_info = ""
    for gene1, gene2 in zip(*genes):
        alignment_info += "   " + (f"[cid={gene1.id}, src={gene1.from_node.id}, dest={gene1.to_node.id}]"
                                   if gene1 is not None else 11 * " " + "-" + 10 * " ")
        alignment_info += "  |  " + (f"[cid={gene2.id}, src={gene2.from_node.id}, dest={gene2.to_node.id}]"
                                     if gene2 is not None else 11 * " " + "-" + 11 * " ") + "\n"

    print(
        "\n\n" + 50 * "#" + "\n\n"
        f"Error while adding the connection {c.from_node.id, c.to_node.id} to a new child node generated by mating.\n"
        f"Parent 1's connections: "
        f"{[(con.from_node.id, con.to_node.id, con.enabled) for con in gen1.connections]}\n"
        f"Parent 2's connections: "
        f"{[(con.from_node.id, con.to_node.id, con.enabled) for con in gen2.connections]}\n"
        f"Child's connections: "
        f"{[(con.from_node.id, con.to_node.id, con.enabled) for con in new_gen.connections]}\n"
        f"Genes alignment: \n{alignment_info}\n"
    )

    gen1.visualize(block_thread=False)
    gen2.visualize()


class ConnectionExistsError(Exception):
    """ Exception that indicates that a connection between two given nodes already exists. """
    pass


class ConnectionToBiasNodeError(Exception):
    """
    Exception that indicates an attempt has been made to create a connection containing a bias node as destination.
    """
    pass


class GenomeIdException(Exception):
    """ Indicates that an attempt has been made to assign an ID to a genome that already has an ID. """
    pass
