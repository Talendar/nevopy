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

""" Implements the genome and its main operations.

A genome is a collection of genes that encode a neural network (the genome's
phenotype). In this implementation, there is no distinction between a genome and
the network it encodes. In NEAT, the genome is the entity subject to evolution.
"""

from typing import Any, cast, Dict, Optional, Sequence, Tuple, List

import numpy as np
np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) \
    # pylint: disable=wrong-import-position

import matplotlib.pyplot as plt
import networkx as nx
from tensorflow import reshape

import nevopy.activations as activations
import nevopy.utils as utils
from nevopy.base_genome import BaseGenome
from nevopy.base_genome import InvalidInputError
from nevopy.base_genome import IncompatibleGenomesError
from nevopy.neat.genes import align_connections
from nevopy.neat.genes import ConnectionGene
from nevopy.neat.genes import NodeGene
from nevopy.neat.config import NeatConfig
from nevopy.neat.id_handler import IdHandler
from nevopy.fixed_topology.genomes import FixedTopologyGenome


class NeatGenome(BaseGenome):
    """ Linear representation of a neural network's connectivity.

    In the context of NEAT, a genome is a collection of genes that encode a
    neural network (the genome's phenotype). In this implementation, there is no
    distinction between a genome and the network it encodes. A genome processes
    inputs based on its nodes and connections in order to produce an output,
    emulating a neural network.

    Note:
        The instances of this class are the entities subject to evolution by the
        NEAT algorithm.

    Note:
        The encoded networks are Graph Neural Networks (GNNs), connectionist
        models that capture the dependence of graphs via message passing between
        the nodes of graphs.

    Note:
        When declaring a subclass of this class, you should always override the
        methods :meth:`.simple_copy()`, :meth:`deep_copy()` and
        :meth:`random_copy()`, so that they return an instance of your subclass
        and not of :class:`.NeatGenome`. It's recommended (although optional) to
        also override the methods :meth:`.distance()` and :meth:`.mate()`.

    Args:
        num_inputs (int): Number of input nodes in the network.
        num_outputs (int): Number of output nodes in the network.
        config (NeatConfig): Settings of the current evolution session.
        initial_connections (bool): If True, connections between the input nodes
            and the output nodes of the network will be created.

    Attributes:
        species_id (int): Indicates the species to which the genome belongs.
        fitness (float): The last calculated fitness of the genome.
        adj_fitness (float): The last calculated adjusted fitness of the genome.
        hidden_nodes (:obj:`list` of :obj:`.NodeGene`): List with all the node
            genes of the type :attr:`.NodeGene.Type.HIDDEN` in the genome.
        connections (:obj:`list` of :obj:`.ConnectionGene`): List with all the
            connection genes in the genome.
        _existing_connections_dict (Dict[int, Set]): Used as a fast lookup table
            to consult existing connections in the network. Given a node N, it
            maps N's ID to the IDs of all the nodes that have a connection with
            N as the source.
    """

    def __init__(self,
                 num_inputs: int,
                 num_outputs: int,
                 config: NeatConfig,
                 initial_connections: bool = True) -> None:
        super().__init__()
        self._config = config
        self.species_id = None        # type: Optional[int]
        self._activated_nodes = None  # type: Optional[Dict[int, bool]]

        self.fitness = 0.0
        self.adj_fitness = 0.0

        self._input_nodes = []
        self.hidden_nodes = []  # type: List[NodeGene]
        self._output_nodes = []
        self._bias_node = None

        self.connections = []  # type: List[ConnectionGene]
        self._existing_connections_dict = {} \
            # type: Dict[int, Dict[int, ConnectionGene]]
        self._output_activation = self.config.out_nodes_activation
        self._hidden_activation = self.config.hidden_nodes_activation

        # init input nodes
        node_counter = 0
        for _ in range(num_inputs):
            self._input_nodes.append(
                NodeGene(node_id=node_counter,
                         node_type=NodeGene.Type.INPUT,
                         activation_func=activations.linear,
                         initial_activation=self.config.initial_node_activation)
            )
            node_counter += 1

        # init bias node
        if self.config.bias_value is not None:
            self._bias_node = NodeGene(
                node_id=node_counter,
                node_type=NodeGene.Type.BIAS,
                activation_func=activations.linear,
                initial_activation=self.config.bias_value,
            )
            node_counter += 1

        # init output nodes
        connection_counter = 0
        for _ in range(num_outputs):
            out_node = NodeGene(
                node_id=node_counter,
                node_type=NodeGene.Type.OUTPUT,
                activation_func=self._output_activation,
                initial_activation=self.config.initial_node_activation,
            )
            self._output_nodes.append(out_node)
            node_counter += 1

            # connecting all input nodes to all output nodes
            if initial_connections:
                for in_node in self._input_nodes:
                    connection_counter += 1
                    self.add_connection(connection_counter, in_node, out_node)

    @property
    def input_shape(self) -> int:
        """ Number of input nodes in the genome. """
        return len(self._input_nodes)

    @property
    def output_shape(self) -> int:
        """ Number of output nodes in the genome. """
        return len(self._output_nodes)

    @property
    def config(self) -> Any:
        return self._config

    @config.setter
    def config(self, c) -> None:
        self._config = c

    def reset_activations(self) -> None:
        """ Resets cached activations of the genome's nodes.

        It restores the current activation value of all the nodes in the network
        to their initial value.
        """
        self._activated_nodes = None
        for n in self.nodes():
            n.reset_activation()

    def reset(self) -> None:
        """ Wrapper for :meth:`.reset_activations`. """
        self.reset_activations()

    def distance(self, other: "NeatGenome") -> float:
        """ Calculates the distance between two genomes.

        The shorter the distance between two genomes, the greater the similarity
        between them is. In the context of NEAT, the similarity between genomes
        increases as:

            1) the number of matching connection genes increases;
            2) the absolute difference between the matching connections weights
               decreases;

        The distance between genomes is used for speciation and for sexual
        reproduction (mating).

        The formula used is shown below. It's the same as the one presented in
        the original NEAT paper :cite:`stanley:ec02`. All the coefficients are
        configurable.

        .. math::
                \\delta = c_1 \\cdot \\frac{E}{N} + c_2 \\cdot \\frac{D}{N} \\
                    + c_3 \\cdot W
            :label: neat_genome_distance

        Args:
            other (NeatGenome): The other genome (an instance of
                :class:`.NeatGenome` or one of its subclasses).

        Returns:
            The distance between the genomes.
        """
        genes = align_connections(self.connections, other.connections)
        excess = disjoint = num_matches = 0
        weight_diff = 0.0

        g1_max_innov = np.amax([c.id for c in self.connections])
        g2_max_innov = np.amax([c.id for c in other.connections])

        for cn1, cn2 in zip(*genes):
            # non-matching genes:
            if cn1 is None or cn2 is None:
                # if c1 is None, c2 can't be None (and vice-versa)
                # noinspection PyUnresolvedReferences
                if ((cn1 is None and cn2.id > g1_max_innov)
                        or (cn2 is None and cn1.id > g2_max_innov)):
                    excess += 1
                else:
                    disjoint += 1
            # matching genes:
            else:
                num_matches += 1
                weight_diff += abs(cn1.weight - cn2.weight)

        c1 = self.config.excess_genes_coefficient
        c2 = self.config.disjoint_genes_coefficient
        c3 = self.config.weight_difference_coefficient

        n = max(len(self.connections), len(other.connections))
        return (((c1 * excess + c2 * disjoint) / n)
                + c3 * weight_diff / num_matches)

    def connection_exists(self, src_id: int, dest_id: int) -> bool:
        """ Checks whether a connection between the given nodes exists.

        Args:
            src_id (int): ID of the connection's source node.
            dest_id (int): ID of the connection's destination node.

        Returns:
            `True` if the specified connection exists in the genome's network
            and `False` otherwise.
        """
        try:
            return dest_id in self._existing_connections_dict[src_id]
        except KeyError:
            return False

    def add_connection(self,
                       cid: int,
                       src_node: NodeGene,
                       dest_node: NodeGene,
                       enabled: bool = True,
                       weight: Optional[float] = None) -> None:
        """ Adds a new connection gene to the genome.

        Args:
            cid (int): ID of the connection. It's used as a historical marker
                of the connection's creation, acting as an "innovation number".
            src_node (NodeGene): Node from where the connection leaves (source
                node).
            dest_node (NodeGene): Node to where the connection is headed
                (destination node).
            enabled (bool): Whether the new connection should be enabled or not.
            weight (Optional[float]): The weight of the connection. If `None`, a
                random value (within the interval specified in the settings)
                will be chosen.

        Raises:
            ConnectionExistsError: If the connection `src_node->dest_node`
                already exists in the genome.
            ConnectionToBiasNodeError: If `dest_node` is an input or bias node
                (nodes of these types do not process inputs!).
        """
        if self.connection_exists(src_node.id, dest_node.id):
            raise ConnectionExistsError(
                f"Attempt to create an already existing connection "
                f"({src_node.id}->{dest_node.id}).")
        if (dest_node.type == NodeGene.Type.BIAS
                or dest_node.type == NodeGene.Type.INPUT):
            raise ConnectionToBiasNodeError(
                f"Attempt to create a connection pointing to a bias or input "
                f"node ({src_node.id}->{dest_node.id}). Nodes of this type "
                f"don't process input.")

        weight = (np.random.uniform(*self.config.new_weight_interval)
                  if weight is None else weight)
        connection = ConnectionGene(cid=cid,
                                    from_node=src_node,
                                    to_node=dest_node,
                                    weight=weight)

        connection.enabled = enabled
        self.connections.append(connection)
        src_node.out_connections.append(connection)
        dest_node.in_connections.append(connection)

        if src_node.id not in self._existing_connections_dict:
            self._existing_connections_dict[src_node.id] = {}
        self._existing_connections_dict[src_node.id][dest_node.id] = connection

    def add_random_connection(self,
                              id_handler: IdHandler,
    ) -> Optional[Tuple[NodeGene, NodeGene]]:
        """  Adds a new connection between two random nodes in the genome.

        This is an implementation of the `add connection mutation`, described in
        the original NEAT paper :cite:`stanley:ec02`.

        Args:
            id_handler (IdHandler): ID handler that will be used to assign an ID
                to the new connection. The handler's internal cache of existing
                connections will be updated accordingly.

        Returns:
            A tuple containing the source node and the destination node of the
            connection, if a new connection was successfully created. `None`, if
            there is no space in the genome for a new connection.
        """
        all_src_nodes = self.nodes()
        np.random.shuffle(all_src_nodes)

        all_dest_nodes = [n for n in all_src_nodes
                          if (n.type != NodeGene.Type.BIAS
                              and n.type != NodeGene.Type.INPUT)]
        np.random.shuffle(all_dest_nodes)

        for src_node in all_src_nodes:
            for dest_node in all_dest_nodes:
                if src_node != dest_node or self.config.allow_self_connections:
                    if not self.connection_exists(src_node.id, dest_node.id):
                        cid = id_handler.next_connection_id(src_node.id,
                                                            dest_node.id)
                        self.add_connection(cid, src_node, dest_node)
                        return src_node, dest_node
        return None

    def enable_random_connection(self) -> None:
        """ Randomly activates a disabled connection gene. """
        disabled = [c for c in self.connections if not c.enabled]
        if len(disabled) > 0:
            connection = np.random.choice(disabled)
            connection.enabled = True

    def add_random_hidden_node(self,
                               id_handler: IdHandler) -> Optional[NodeGene]:
        """ Adds a new hidden node to the genome in a random position.

        This method implements the `add node mutation` procedure described in
        the original NEAT paper:

        "An existing connection is split and the new node placed where the old
        connection used to be. The old connection is disabled and two new
        connections are added to the genome. The new connection leading into the
        new node receives a weight of 1, and the new connection leading out
        receives the same weight as the old connection." - :cite:`stanley:ec02`

        Only currently enabled connections are considered eligible to "host" the
        new hidden node.

        Args:
            id_handler (IdHandler): ID handler that will be used to assign an ID
                to the new hidden node. The handler's internal cache of existing
                nodes and connections will be updated accordingly.

        Returns:
            The new hidden node, if it was successfully created. `None` if it
            wasn't possible to find a connection to "host" the new node. This
            usually happens when the ID handler hasn't been reset in a while.
        """
        eligible_connections = [c for c in self.connections if c.enabled]
        if not eligible_connections:
            return None
        np.random.shuffle(eligible_connections)

        for original_connection in eligible_connections:
            src_node = original_connection.from_node
            dest_node = original_connection.to_node

            hid = id_handler.next_hidden_node_id(src_node.id, dest_node.id)
            if (self.connection_exists(src_node.id, hid)
                    or self.connection_exists(hid, dest_node.id)):
                # might happen if the id handler cache hasn't been reset yet
                continue

            original_connection.enabled = False
            new_node = NodeGene(
                node_id=hid,
                node_type=NodeGene.Type.HIDDEN,
                activation_func=self._hidden_activation,
                initial_activation=self.config.initial_node_activation
            )
            self.hidden_nodes.append(new_node)

            # adding connections
            cid = id_handler.next_connection_id(src_node.id, new_node.id)
            self.add_connection(cid,
                                src_node, new_node,
                                weight=1)

            cid = id_handler.next_connection_id(new_node.id, dest_node.id)
            self.add_connection(cid,
                                new_node, dest_node,
                                weight=original_connection.weight)
            return new_node

        return None

    def mutate_weights(self) -> None:
        """ Randomly mutates the weights of the genome's connections.

        Each connection gene in the genome has a chance to be perturbed, reset
        or to remain unchanged.
        """
        for connection in self.connections:
            if utils.chance(self.config.weight_reset_chance):
                # perturbating the connection
                connection.weight = np.random.uniform(
                    *self.config.new_weight_interval)
            else:
                # resetting the connection
                p = np.random.uniform(low=-self.config.weight_perturbation_pc,
                                      high=self.config.weight_perturbation_pc)
                d = connection.weight * p
                connection.weight += d

    def simple_copy(self) -> "NeatGenome":
        """ Makes a simple copy of the genome.

        Wraps a call to this class' constructor.

        Returns:
            A copy of the genome without any of its connections (including the
            ones between input and output nodes) and hidden nodes.
        """
        return NeatGenome(num_inputs=len(self._input_nodes),
                          num_outputs=len(self._output_nodes),
                          config=self.config,
                          initial_connections=False)

    def __copy_aux(self, random_weights: bool) -> "NeatGenome":
        """ Auxiliary function used for deep copying the genome, with or without
        random weights.
        """
        new_genome = self.simple_copy()
        copied_nodes = {n.id: n for n in new_genome.nodes()}

        # creating required nodes
        for node in self.hidden_nodes:
            new_node = node.simple_copy()
            copied_nodes[node.id] = new_node
            new_genome.hidden_nodes.append(new_node)

        # adding connections
        for c in self.connections:
            try:
                new_genome.add_connection(
                    cid=c.id,
                    src_node=copied_nodes[c.from_node.id],
                    dest_node=copied_nodes[c.to_node.id],
                    enabled=c.enabled,
                    weight=c.weight if not random_weights else None)
            except ConnectionExistsError as e:
                cons = [f"[{con.id}] {con.from_node.id}->{con.to_node.id} "
                        f"({con.enabled})" for con in self.connections]
                raise ConnectionExistsError(
                    "Connection exists error when duplicating parent.\n"
                    f"Source node's connections: {cons}") from e

        return new_genome

    def random_copy(self) -> "NeatGenome":
        """ Makes a deep copy of the genome, but with random weights.

        Returns:
            A deep copy of the genome with the same topology of the original
            genome, but random connections weights.
        """
        return self.__copy_aux(random_weights=True)

    def deep_copy(self) -> "NeatGenome":
        """ Makes an exact/deep copy of the genome.

        All the nodes and connections (including their weights) of the parent
        genome are copied to the new genome.

        Returns:
            An exact/deep copy of the genome.
        """
        return self.__copy_aux(random_weights=False)

    def process_node(self, n: NodeGene) -> float:
        """ Recursively processes the activation of the given node.

        Unless it's a bias or input node (that have a fixed output), a node must
        process the input it receives from other nodes in order to produce an
        activation. This is done recursively: if `n` receives input from a node
        `m` that haven't had its activation calculated yet, the activation of
        `m` will be calculated recursively before the activation of `n` is
        computed. Recurrences are solved by using the previous activation of the
        "problematic" node.

        Let :math:`w_i` be the weight of the :math:`i^{\\text{th}}` connection
        that has `n` as destination node. Let :math:`a_i` be the current cached
        output of the source node of :math:`c_i`. Let :math:`\\sigma` be the
        activation function of `n`. The activation (output) `a` of `n` is
        computed as follows:

        :math:`a = \\sigma (\\sum \\limits_{i} w_i \\cdot a_i)`

        Args:
            n (NodeGene): The node to be processed.

        Returns:
            The activation value (output) of the node.
        """
        # checking if the node needs to be activated
        if (n.type != NodeGene.Type.INPUT
                and n.type != NodeGene.Type.BIAS
                and not self._activated_nodes[n.id]):
            # activating the node
            # the current node (n) is immediately marked as activated; this is
            # needed due to recurrency: if, during the recursive calls, some
            # node m depends on the activation of n, the old activation of n
            # is used.
            self._activated_nodes[n.id] = True
            zsum = 0.0
            for connection in n.in_connections:
                if connection.enabled:
                    src_node, weight = connection.from_node, connection.weight
                    zsum += weight * self.process_node(src_node)
            n.activate(zsum)
        return n.activation

    def process(self, x: Sequence[float]) -> np.ndarray:
        """ Feeds the given input to the neural network.

        In this implementation, there is no distinction between a genome and
        the neural network it encodes. The genome will emulate a neural network
        (its phenotype) in order to process the given input. The encoded network
        is a Graph Neural Networks (GNN).

        Note:
            The processing is done recursively, starting from the output nodes
            (top-down approach). Because of that, nodes not connected to at
            least one of the network's output nodes won't be processed.

        Args:
            x (Sequence[float]): A sequence object (like a list or numpy array)
                containing the inputs to be fed to the neural network input
                nodes. It represents a single training sample. The value in the
                index `i` of `X` will be fed to the :math:`i^{th}` input node
                of the neural network.

        Returns:
            A numpy array containing the outputs of the network's output nodes.
            The index `i` contains the activation value of the :math:`i^{th}`
            output node of the network.

        Raises:
            InvalidInputError: If the number of elements in `X` doesn't match
                the number of input nodes in the network.
        """
        if len(x) != len(self._input_nodes):
            raise InvalidInputError(
                "The input size must match the number of input nodes in the "
                f"network! Expected input of length {len(self._input_nodes)} "
                f"but got {len(x)}."
            )

        # preparing input nodes
        for in_node, value in zip(self._input_nodes, x):
            in_node.activate(value)

        # resetting activated nodes dict
        self._activated_nodes = {
            n.id: False
            for n in self._output_nodes + self.hidden_nodes
        }

        # processing nodes in a top-down manner (starts from the output nodes)
        # nodes not connected to at least one output node are not processed
        h = np.zeros(len(self._output_nodes))
        for i, out_node in enumerate(self._output_nodes):
            h[i] = self.process_node(out_node)

        return h

    def nodes(self) -> List[NodeGene]:
        """
        Returns all the genome's node genes. Order: inputs, bias, outputs and
        hidden.
        """
        return (self._input_nodes +
                ([self._bias_node] if self._bias_node is not None else []) +
                self._output_nodes +
                self.hidden_nodes)

    def valid_out_nodes(self) -> bool:
        """ Checks if all the genome's output nodes are valid.

        An output node is considered to be valid if it receives, during its
        processing, at least one input, i.e., the node has at least one enabled
        incoming connection. Invalid output nodes simply outputs a fixed
        default value and are, in many cases, undesirable.

        Returns:
            `True` if all the genome's output nodes have at least one enabled
            incoming connection and `False` otherwise. Self-connecting
            connections are not considered.
        """
        for out_node in self._output_nodes:
            valid = False
            for in_con in out_node.in_connections:
                if in_con.enabled and not in_con.self_connecting():
                    valid = True
                    break
            if not valid:
                return False
        return True

    def valid_in_nodes(self) -> bool:
        """ Checks if all the genome's input nodes are valid.

        An input node is considered to be valid if it has at least one enabled
        connection leaving it, i.e., its activation is used as input by at least
        one other node.

        Returns:
            `True` if all the genome's input nodes are valid and `False`
            otherwise.
        """
        for in_node in self._input_nodes:
            valid = False
            for out_con in in_node.out_connections:
                if out_con.enabled:
                    valid = True
                    break
            if not valid:
                return False
        return True

    def mate(self, other: "NeatGenome") -> "NeatGenome":
        """ Mates two genomes to produce a new genome (offspring).

        Sexual reproduction. Follows the idea described in the original paper of
        the NEAT algorithm:

        "When crossing over, the genes in both genomes with the same innovation
        numbers are lined up. These genes are called matching genes. (...).
        Matching genes are inherited randomly, whereas disjoint genes (those
        that do not match in the middle) and excess genes (those that do not
        match in the end) are inherited from the more fit parent. (...) [If the
        parents fitness are equal] the disjoint and excess genes are also
        inherited randomly. (...) there’s a preset chance that an inherited gene
        is disabled if it is disabled in either parent." - :cite:`stanley:ec02`

        Args:
            other (NeatGenome): The second genome. Currently,
                :class:`.NeatGenome` is only compatible for mating with
                instances of :class:`.NeatGenome` or of one of its subclasses.

        Returns:
            A new genome (the offspring born from the sexual reproduction
            between the current genome and the genome passed as argument.

        Raises:
            IncompatibleGenomesError: If the genome passed as argument to
                ``other`` is incompatible with the current genome (`self`).
        """
        if not issubclass(type(other), NeatGenome):
            raise IncompatibleGenomesError(
                "Instances of `NeatGenome` are currently only compatible for "
                "sexual reproduction with instances of `NeatGenome or one of "
                "its subclasses!"
            )

        # aligning matching genes
        genes = align_connections(self.connections, other.connections)

        # new genome
        new_gen = self.simple_copy()
        copied_nodes = {n.id: n for n in new_gen.nodes()}

        # mate (choose new genome's connections)
        chosen_connections = []
        for c1, c2 in zip(*genes):
            if c1 is None and self.adj_fitness > other.adj_fitness:
                # case 1: the gene is missing on self and self is dominant
                # (higher fitness); action: ignore the gene
                continue

            if c2 is None and other.adj_fitness > self.adj_fitness:
                # case 2: the gene is missing on other and other is dominant
                # (higher fitness); action: ignore the gene
                continue

            # case 3: the gene is missing either on self or on other and their
            # fitness are equal; action: random choice

            # case 4: the gene is present both on self and on other; action:
            # random choice

            c = np.random.choice((c1, c2))
            if c is not None:
                # if the gene is disabled in either parent, it has a chance to
                # also be disabled in the new genome
                enabled = True
                if ((c1 is not None and not c1.enabled)
                        or (c2 is not None and not c2.enabled)):
                    enabled = not utils.chance(
                        self.config.disable_inherited_connection_chance)
                chosen_connections.append((c, enabled))

                # adding the hidden nodes of the connection (if needed)
                for node in (c.from_node, c.to_node):
                    if (node.type == NodeGene.Type.HIDDEN
                            and node.id not in copied_nodes):
                        new_node = node.simple_copy()
                        new_gen.hidden_nodes.append(new_node)
                        copied_nodes[node.id] = new_node

        # adding inherited connections
        for c, enabled in chosen_connections:
            src_node = copied_nodes[c.from_node.id]
            dest_node = copied_nodes[c.to_node.id]
            try:
                new_gen.add_connection(cid=c.id,
                                       src_node=src_node, dest_node=dest_node,
                                       enabled=enabled, weight=c.weight)
            except ConnectionExistsError:
                # if this exception is raised, it means that the connection was
                # already inherited from the other parent; this is possible
                # because, in some cases, a connection between the same two
                # nodes appears in different generations and are assigned,
                # because of that, different IDs.
                pass
                # _debug_mating(genes, c, self, other, new_gen)
                # raise ConnectionExistsError()
        return new_gen

    def info(self) -> str:
        """
        Returns a string with the genome's nodes activations and connections.
        Used mostly for debugging purposes.
        """
        txt = ">> NODES ACTIVATIONS\n"
        for n in self.nodes():
            txt += f"[{n.id}][{str(n.type).split('.')[1][0]}] {n.activation}\n"
        txt += "\n>> CONNECTIONS\n"
        for c in self.connections:
            txt += f"[{'ON' if c.enabled else 'OFF'}][{c.id}]" \
                   f"[{c.from_node.id}->{c.to_node.id}] {c.weight}\n"
        return txt

    def columns_graph_layout(self,
                             width: float,
                             height: float,
                             node_size: int,
                             pad_width_pc: float = 0.025,
                             ideal_h_nodes_per_col: int = 4,
    ) -> Dict[int, Tuple[float, float]]:
        """ Positions the network's nodes in columns.

        The input nodes are placed in the left-most column and the output nodes
        are placed in the right-most columns. The hidden nodes are placed in
        columns located between those two columns. For big networks, try using a
        smaller node size for better quality.

        Args:
            width (float): Width of the `matplotlib` figure, in inches (default
                unit).
            height (float): Height of the `matplotlib` figure, in inches
                (default unit).
            node_size (int): Size of the drawn nodes, in `points**2` (the area
                of each node). See the parameter ``s`` of
                `matplotlib.axes.Axes.scatter
                <https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.axes.Axes.scatter.html>`_
                for more information.
            pad_width_pc (float): Percentage of the figure's width to be
                reserved for padding in each of the figure's border.
            ideal_h_nodes_per_col (int): Preferred number of hidden nodes per
                column (the algorithm will try to draw columns with this amount
                of hidden nodes when possible).

        Returns:
            Dictionary mapping the ID of each node to a tuple containing its
            position in the figure.
        """
        # adjusting limits
        plt.xlim(0, width)
        plt.ylim(0, height)

        # getting real node diameter
        dpi = plt.gcf().dpi
        node_size = (node_size ** 0.5) / dpi

        # padding
        origin = width * pad_width_pc
        width, height = width - 2*origin, height - 2*origin
        pos = {}

        # procedure for inserting nodes into columns
        def insert_nodes_col(x, nodes):
            """ Inserts the nodes in a column (specified by x). """
            if len(nodes) == 1:
                pos[nodes[0].id] = (x, origin + height/2)
                return

            next_y = origin
            space_y = height / len(nodes)
            for n in nodes:
                pos[n.id] = (x, next_y + space_y/2)
                next_y += space_y

        # input and bias nodes
        insert_nodes_col(
            x=origin,
            nodes=self._input_nodes + ([] if self._bias_node is None
                                       else [self._bias_node]),
        )

        # output nodes
        insert_nodes_col(x=origin + width,
                         nodes=self._output_nodes)

        # hidden nodes
        if self.hidden_nodes:
            max_num_h_cols = int((width - 4*node_size) / (2*node_size))
            h_nodes_per_col = ideal_h_nodes_per_col
            num_cols = max(1, np.ceil(len(self.hidden_nodes) / h_nodes_per_col))
            while num_cols > max_num_h_cols:
                h_nodes_per_col += 1
                num_cols = np.ceil(len(self.hidden_nodes) / h_nodes_per_col)

            if num_cols == 1:
                insert_nodes_col(x=width / 2, nodes=self.hidden_nodes)
            else:
                next_x = origin + 2*node_size
                space_x = (width - 4*node_size) / num_cols
                h_nodes = np.array(self.hidden_nodes)
                for node_list in np.array_split(h_nodes, num_cols):
                    insert_nodes_col(x=next_x + space_x/2,
                                     nodes=node_list)
                    next_x += space_x

        return pos

    def visualize(self,
                  layout_name: str = "columns",
                  layout_kwargs: Optional[Dict[str, Any]] = None,
                  show: bool = True,
                  block_thread: bool = True,
                  save_to: Optional[str] = None,
                  save_transparent: bool = False,
                  figsize: Tuple[int, int] = (10, 6),
                  node_size: int = 300,
                  pad: int = 1,
                  legends: bool = True,
                  nodes_ids: bool = True,
                  node_id_color: str = "black",
                  edge_curviness: float = 0.1,
                  edges_ids: bool = False,
                  edge_id_color: str = "black",
                  background_color: str = "snow",
                  legend_box_color: str = "honeydew",
                  input_color: str = "deepskyblue",
                  output_color: str = "mediumseagreen",
                  hidden_color: str = "silver",
                  bias_color: str = "khaki",
                  **kwargs,  # pylint: disable=unused-argument
    ) -> None:
        """ Plots the neural network (phenotype) encoded by the genome.

        The network is drawn as a graph, with nodes and edges. An edge's color
        is chosen according to the edge's weight. Edges with greater weights are
        drawn with more intense / stronger colors. Edges connecting a node to
        itself aren't be drawn.

        This method uses `NetworkX <https://github.com/networkx/networkx>`_ to
        handle the drawings. It positions the network's nodes according to a
        layout, whose name you can specify in the parameter ``layout_name``. The
        currently available layouts are:

        * All the standard `NetworkX's` layouts available in this
          `link
          <https://networkx.org/documentation/latest/reference/drawing.html#module-networkx.drawing.layout>`_;
        * The `graphviz` layout; it's really good, but to use it you must
          have `Graphviz-Dev` and `pygraphviz` installed on your machine;
        * The `columns` layout (used by default), implemented exclusively for
          `NEvoPy`; it positions the nodes in columns (see
          :meth:`.NeatGenome.columns_graph_layout`, specially the parameter
          ``ideal_h_nodes_per_col``).

        For the colors parameters, it's possible to pass a string with the color
        HEX value or a string with the color's name (names available here:
        https://matplotlib.org/3.1.0/gallery/color/named_colors.html).

        Args:
            layout_name (str): The name of the layout to be used to position the
                network's nodes.
            layout_kwargs (Optional[Dict[str, Any]]): Keyed arguments to be
                passed to the layout. Check each layout documentation for more
                information about the accepted arguments.
            show (bool): Whether to show the generated image or not. If True, a
                window will be created by `matplotlib` to show the image.
            block_thread (bool): Whether to block the execution's thread while
                showing the image. Useful for visualizing multiple networks at
                once. In this case, you should call :meth:`.visualize` with this
                parameter set to `False` on all genomes except for the last one,
                so all the windows are shown simultaneously.
            save_to (Optional[str]): Path to save the image. If `None`, the
                image won't be automatically saved.
            save_transparent: Whether the saved image should have a transparent
                background or not.
            figsize (Tuple[int, int]): Size of the matplotlib figure.
            node_size (int): Size of the drawn nodes, in `points**2` (the area
                of each node). Default size is 300. See the parameter ``s`` of
                `matplotlib.axes.Axes.scatter
                <https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.axes.Axes.scatter.html>`_
                for more information.
            pad (int): The image's padding (distance between the figure of the
                network and the image's border).
            legends (bool): If `True`, a box with legends describing the nodes
                colors will be drawn.
            nodes_ids (bool): If `True`, the nodes will have their ID drawn
                inside them.
            node_id_color (str): Color of the drawn nodes ids.
            edge_curviness (float): Angle, in radians, of the edges arcs. A
                value of 0 indicates a straight line.
            edges_ids (bool): If `True`, each connection/edge will have its ID
                drawn on it. Keep in mind that some labels might overlap with
                each other, making only one of them visible.
            edge_id_color (str): Color of the drawn connections/edges ids.
            background_color (str): Color of the figure's background.
            legend_box_color (str): Color of the legend box.
            input_color (str): Color of the input nodes.
            output_color (str): Color of the output nodes.
            hidden_color (str): Color of the hidden nodes.
            bias_color (str): Color of the bias node.

        Raises:
            RuntimeError: If both `show` and `save_to` parameters are set to
                `False` (in which case the function wouldn't be doing anything
                but wasting computation).
        """
        # validating args
        if not show and not save_to:
            raise RuntimeError("Both \"show\" and \"save_to\" parameters are "
                               "set to False!")

        # config and start
        plt.rcParams["axes.facecolor"] = background_color
        graph = nx.MultiDiGraph()
        graph.add_nodes_from([n.id for n in self.nodes()])
        plt.figure(figsize=figsize)

        if layout_kwargs is None:
            layout_kwargs = {}

        # connections
        edges_labels = {}
        for c in self.connections:
            if c.enabled:
                graph.add_edge(c.from_node.id, c.to_node.id, weight=c.weight)
                edges_labels[(c.from_node.id, c.to_node.id)] = c.id

        # selecting layout
        if layout_name == "graphviz":
            try:
                # pylint: disable=import-outside-toplevel
                # pylint: disable=unused-import
                import pygraphviz
                from networkx.drawing.nx_agraph import graphviz_layout
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    "Couldn't find the package `pygraphviz`!\nTo draw the "
                    "genome's neural network, this package is required. To "
                    "install it, however, you first need to install the dev "
                    "version of `Graphviz` (https://graphviz.org/download/) on "
                    "your system. On Ubuntu, you can do that by executing the "
                    "following command:\n"
                    "\t~$ sudo apt-get install -y graphviz-dev\n"
                    "After installing `Graphviz`, just use pip to install "
                    "`pygraphviz` and you're all set:\n"
                    "\t~$ pip install pygraphviz") from e

            if "prog" not in layout_kwargs:
                layout_kwargs["prog"] = "dot"
            if "args" not in layout_kwargs:
                layout_kwargs["args"] = "-Grankdir=LR"
            pos = graphviz_layout(graph, **layout_kwargs)

        elif layout_name == "columns":
            pos = self.columns_graph_layout(*figsize, node_size,
                                            **layout_kwargs)
        else:
            nx_layout_func = getattr(nx, layout_name)
            pos = nx_layout_func(graph, **layout_kwargs)

        # plotting
        nx.draw_networkx_nodes(graph, pos=pos,
                               nodelist=[n.id for n in self._input_nodes],
                               node_size=[node_size] * len(self._input_nodes),
                               node_color=input_color, label="Input nodes")
        nx.draw_networkx_nodes(graph, pos=pos,
                               nodelist=[n.id for n in self._output_nodes],
                               node_size=[node_size] * len(self._output_nodes),
                               node_color=output_color, label="Output nodes")
        nx.draw_networkx_nodes(graph, pos=pos,
                               nodelist=[n.id for n in self.hidden_nodes],
                               node_size=[node_size] * len(self.hidden_nodes),
                               node_color=hidden_color, label="Hidden nodes")

        if self._bias_node is not None:
            nx.draw_networkx_nodes(graph, pos=pos,
                                   nodelist=[self._bias_node.id],
                                   node_size=[node_size],
                                   node_color=bias_color, label="Bias node")

        if graph.number_of_edges() > 0:
            # calculating edges colors
            edges_weights = list(nx.get_edge_attributes(graph,
                                                        "weight").values())
            if len(edges_weights) == 1:
                edges_colors = [(1, 0.5, 0, 1)]
            else:
                min_w, max_w = np.min(edges_weights), np.max(edges_weights)
                edges_colors = [(1, 0.6 * (1 - (w - min_w) / (max_w - min_w)),
                                 0, 0.3 + 0.7 * (w - min_w) / (max_w - min_w))
                                for w in edges_weights]

            # drawing edges
            nx.draw_networkx_edges(
                graph,
                pos=pos,
                edge_color=edges_colors,
                connectionstyle=f"arc3, rad={edge_curviness}"
            )

        if edges_ids:
            nx.draw_networkx_edge_labels(graph, pos, edge_labels=edges_labels,
                                         font_color=edge_id_color)

        if nodes_ids:
            nx.draw_networkx_labels(graph, pos,
                                    labels={k.id: k.id
                                            for k in self.nodes()},
                                    font_size=10,
                                    font_color=node_id_color,
                                    font_family="sans-serif")
        if legends:
            plt.legend(facecolor=legend_box_color, borderpad=0.8,
                       labelspacing=0.5)

        plt.tight_layout(pad=pad)
        if save_to is not None:
            plt.savefig(save_to, transparent=save_transparent)

        if show:
            plt.show(block=block_thread)


def _debug_mating(genes, c, gen1, gen2, new_gen):
    """ Used to debug the "mate_genomes" function. """
    alignment_info = ""
    for gene1, gene2 in zip(*genes):
        alignment_info += (
            "   " + (f"[cid={gene1.id}, src={gene1.from_node.id}, "
                     f"dest={gene1.to_node.id}]"
                     if gene1 is not None
                     else 11 * " " + "-" + 10 * " ") +
            "  |  " + (f"[cid={gene2.id}, src={gene2.from_node.id}, "
                       f"dest={gene2.to_node.id}]"
                       if gene2 is not None
                       else 11 * " " + "-" + 11 * " ") +
            "\n"
        )

    p1_cons = [(con.from_node.id, con.to_node.id, con.enabled)
               for con in gen1.connections]
    p2_cons = [(con.from_node.id, con.to_node.id, con.enabled)
               for con in gen2.connections]
    child_cons = [(con.from_node.id, con.to_node.id, con.enabled)
                  for con in new_gen.connections]

    print(
        "\n\n" + 50 * "#" + "\n\n"
        f"Error while adding the connection {c.from_node.id, c.to_node.id} "
        f"to a new child node generated by mating.\n"
        f"Parent 1's connections: {p1_cons}\n"
        f"Parent 2's connections: {p2_cons}\n"
        f"Child's connections: {child_cons}\n"
        f"Genes alignment: \n{alignment_info}\n"
    )

    gen1.visualize(block_thread=False)
    gen2.visualize()


class ConnectionExistsError(Exception):
    """
    Exception that indicates that a connection between two given nodes already
    exists.
    """
    pass


class ConnectionToBiasNodeError(Exception):
    """
    Exception that indicates that an attempt has been made to create a
    connection containing a bias node as destination.
    """
    pass


class FixTopNeatGenome(NeatGenome):
    """ Integration of a NEAT genome with a fixed topology genome.

    This class defines a new type of NEAT genome that integrates the default
    :class:`.NeatGenome with a :class:`.FixedTopologyGenome`. It can be used
    with :class:`.NeatPopulation`.

    When an input is received, it's first processed by the layers of the fixed
    topology genome. The output is, then, processed using NEAT, which generates
    the final output.

    Note:
        This class is useful when the inputs that will be fed to the genome have
        high dimensions. Since NEAT doesn't scale well with such lengthy inputs
        (like images), a fixed topology genome (that can contain, for instance,
        convolutional layers) can be used to reduce the dimensionality of the
        input before feeding it to NEAT's nodes.

    Args:
        fito_genome (FixedTopologyGenome): Instance of
            :class:`.FixedTopologyGenome` to be used to pre-process the inputs.
            It will also be evolved.
        num_neat_inputs (int): Length of the flattened outputs of the fixed
            topology genome. It's also the number of input nodes of the NEAT
            genome.
        num_neat_outputs (int): Number of output nodes of the NEAT genome.
        config (NeatConfig): Settings of the current evolutionary session.
        initial_neat_connections (bool): Whether to create connections
            connecting each input node of the NEAT genome to each of its output
            nodes.
    """

    def __init__(self,
                 fito_genome: FixedTopologyGenome,
                 num_neat_inputs: int,
                 num_neat_outputs: int,
                 config: NeatConfig,
                 initial_neat_connections: bool = True) -> None:
        super().__init__(num_inputs=num_neat_inputs,
                         num_outputs=num_neat_outputs,
                         config=config,
                         initial_connections=initial_neat_connections)
        self.fito_genome = fito_genome

    def distance(self, other: NeatGenome) -> float:
        """ Sums, to the default distance calculated by
        :meth:`.NeatGenome.distance()`, the sum of the absolute difference
        between the fixed topology layers weights.
        """
        dist = super().distance(other)
        extra_dist = 0.0
        if isinstance(other, FixTopNeatGenome):
            total_weights = 0
            for l1, l2 in zip(self.fito_genome.layers,
                              other.fito_genome.layers):
                for w1, w2 in zip(l1.weights, l2.weights):
                    extra_dist += np.abs(w1 - w2).sum()
                    total_weights += w1.size

            extra_dist *= self.config.weight_difference_coefficient
            if total_weights > 0:
                extra_dist /= total_weights

        return dist + extra_dist

    def mutate_weights(self) -> None:
        super().mutate_weights()
        if self.fito_genome.config.maex_counter != self.config.maex_counter:
            self.fito_genome.config.update_mass_extinction(
                self.config.maex_counter,
            )
        self.fito_genome.mutate_weights()

    def simple_copy(self) -> "FixTopNeatGenome":
        """ Makes a simple copy of the genome.

        Wraps a call to this class' constructor. The new genome's is initialized
        without a fixed topology genome (:attr:`.fito_genome`) - the value of
        this attribute is `None`.

        Returns:
            A copy of the genome without any of its connections (including the
            ones between input and output nodes) and hidden nodes. The attribute
            :attr:`.fito_genome` is set to None.
        """
        return FixTopNeatGenome(fito_genome=None,
                                num_neat_inputs=self.input_shape,
                                num_neat_outputs=self.output_shape,
                                config=self.config,
                                initial_neat_connections=False)

    def random_copy(self) -> "FixTopNeatGenome":
        new_genome = super().random_copy()
        new_genome = cast(FixTopNeatGenome, new_genome)
        new_genome.fito_genome = self.fito_genome.random_copy()
        return new_genome

    def deep_copy(self) -> "FixTopNeatGenome":
        new_genome = super().deep_copy()
        new_genome = cast(FixTopNeatGenome, new_genome)
        new_genome.fito_genome = self.fito_genome.deep_copy()
        return new_genome

    def process(self, x: Sequence[float]) -> np.ndarray:
        """ Feeds the input to the fixed topology genome and uses the output as
        input to the NEAT genome.
        """
        x = reshape(self.fito_genome.process(x), [-1])
        return super().process(x)

    def mate(self, other: NeatGenome) -> NeatGenome:
        new_genome = super().mate(other)
        if self.fito_genome.config.maex_counter != self.config.maex_counter:
            self.fito_genome.config.update_mass_extinction(
                self.config.maex_counter
            )

        if isinstance(other, FixTopNeatGenome):
            new_genome = cast(FixTopNeatGenome, new_genome)
            new_genome.fito_genome = self.fito_genome.mate(other.fito_genome)

        return new_genome
