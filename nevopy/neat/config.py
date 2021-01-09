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

""" This module implements the :class:`Config` class, used to handle the
settings of the NEAT algorithm.
"""

import nevopy


class Config:
    """ Stores the settings of the NEAT algorithm.

    The configurations can be ignored (default values will be used), set in the
    arguments of this class constructor or written in a file (pathname passed
    as an argument).

    Todo: specify the config file format.

    Args:
        file_pathname (Optional[str]): The pathname of a file from where the
            settings should be loaded.
        weight_mutation_chance (float): Chance of mutating a connection gene.
        weight_perturbation_pc (float): Maximum absolute percentage value for
            the perturbation of the weights.
        weight_reset_chance (float): Chance of resetting the weight of a
            connection gene (assign it a new random value).
        new_weight_interval (Tuple[float, float]): An interval of values that a
            new random weight of a connection gene can have.
        excess_genes_coefficient (float): Used in the formula to calculate the
            distance between two genomes. It's the :math:`c_1` coefficient of
            :eq:`distance`.
        disjoint_genes_coefficient (float): Used in the formula to calculate the
            distance between two genomes. It's the :math:`c_2` coefficient of
            :eq:`distance`.
        weight_difference_coefficient: Used in the formula to calculate the
            distance between two genomes. It's the :math:`c_3` coefficient of
            :eq:`distance`.
        species_distance_threshold (float): Minimum distance, as calculated by
            :eq:`distance`, between two genomes for them to be considered as
            being of the same species. A lower threshold will make new species
            easier to appear, increasing the number of species throughout the
            evolution process.
        species_elitism_threshold (int): Species with a number of members
            superior to this threshold will have their fittest member copied
            unchanged to the next generation.
        weak_genomes_removal_pc (float): Percentage of the least fit individuals
            to be deleted from the population before the reproduction step.
        mating_chance (float): Chance for a genome to reproduce sexually, i.e.,
            by mating / crossing-over with another genome. Decreasing this value
            increases the chance of a genome reproducing asexually, through
            binary fission (copy + mutation).
        interspecies_mating_chance (float): Chance for a sexual reproduction
            (mating / cross-over) to be between genomes of different species.
        new_node_mutation_chance (float): Chance of a new hidden node being
            added to a newly born genome.
        new_connection_mutation_chance (float): Chance of a new connection being
            added to a newly born genome.
        enable_connection_mutation_chance (float): Chance of enabling a disabled
            connection in a newly born genome.
        rank_prob_dist_coefficient (float): Coefficient :math:`\\alpha` used to
            calculate the probability distribution used to select genomes for
            reproduction. Basically, the value of this constant can be
            interpreted as follows: the genome, within a species, with the
            highest fitness has :math:`\\times \\alpha` more chance of being
            selected for reproduction than the second best genome, which, in
            turn, has :math:`\\times \\alpha` more chance of being selected than
            the third best genome, and so forth. This approach to reproduction
            is called rank-based selection. Note that this is applied to
            individuals within the same species.
        out_nodes_activation (Callable[[float], float]): Activation function to
            be used by the output nodes of the networks. It should receive a
            float as input and return a float (the resulting activation) as
            output.
        hidden_nodes_activation (Callable[[float], float]): Activation function
            to be used by the hidden nodes of the networks. It should receive a
            float as input and return a float (the resulting activation) as
            output.
        bias_value (Optional[float]): Constant activation value to be used by
            the bias nodes. If `None`, bias nodes won't be used.
        reset_innovations_period (Optional[int]): If None, the innovation IDs of
            the new genes will never be reset. If an int, the innovation IDs
            will be reset with a period (number of generations passed) equal to
            the value specified. As long as the id handler isn't reset, a hidden
            node can't be inserted more than once in a connection between two
            given nodes.
        allow_self_connections (bool): Whether to allow connections connecting a
            node to itself being formed. If a node is connected to itself, it
            considers its last output when calculating its new output.
        disable_inherited_connection_chance (bool): During a sexual reproduction
            between two genomes, this constant specifies the chance of a
            connection in the newly born genome being disabled if it's disabled
            on at least one of the parent genomes.
        initial_node_activation (float): Initial activation value cached by a
            node when it's created or reset.

    Attributes:
        file_pathname (Optional[str]): The pathname of a file from where the
            settings should be loaded.
        weight_mutation_chance (float): Chance of mutating a connection gene.
        weight_perturbation_pc (float): Maximum absolute percentage value for
            the perturbation of the weights.
        weight_reset_chance (float): Chance of resetting the weight of a
            connection gene (assign it a new random value).
        new_weight_interval (Tuple[float, float]): An interval of values that a
            new random weight of a connection gene can have.
        excess_genes_coefficient (float): Used in the formula to calculate the
            distance between two genomes. It's the :math:`c_1` coefficient of
            :eq:`distance`.
        disjoint_genes_coefficient (float): Used in the formula to calculate the
            distance between two genomes. It's the :math:`c_2` coefficient of
            :eq:`distance`.
        weight_difference_coefficient: Used in the formula to calculate the
            distance between two genomes. It's the :math:`c_3` coefficient of
            :eq:`distance`.
        species_distance_threshold (float): Minimum distance, as calculated by
            :eq:`distance`, between two genomes for them to be considered as
            being of the same species. A lower threshold will make new species
            easier to appear, increasing the number of species throughout the
            evolution process.
        species_elitism_threshold (int): Species with a number of members
            superior to this threshold will have their fittest member copied
            unchanged to the next generation.
        weak_genomes_removal_pc (float): Percentage of the least fit individuals
            to be deleted from the population before the reproduction step.
        mating_chance (float): Chance for a genome to reproduce sexually, i.e.,
            by mating / crossing-over with another genome. Decreasing this value
            increases the chance of a genome reproducing asexually, through
            binary fission (copy + mutation).
        interspecies_mating_chance (float): Chance for a sexual reproduction
            (mating / cross-over) to be between genomes of different species.
        new_node_mutation_chance (float): Chance of a new hidden node being
            added to a newly born genome.
        new_connection_mutation_chance (float): Chance of a new connection being
            added to a newly born genome.
        enable_connection_mutation_chance (float): Chance of enabling a disabled
            connection in a newly born genome.
        rank_prob_dist_coefficient (float): Coefficient :math:`\\alpha` used to
            calculate the probability distribution used to select genomes for
            reproduction. Basically, the value of this constant can be
            interpreted as follows: the genome, within a species, with the
            highest fitness has :math:`\\times \\alpha` more chance of being
            selected for reproduction than the second best genome, which, in
            turn, has :math:`\\times \\alpha` more chance of being selected than
            the third best genome, and so forth. This approach to reproduction
            is called rank-based selection. Note that this is applied to
            individuals within the same species.
        out_nodes_activation (Callable[[float], float]): Activation function to
            be used by the output nodes of the networks. It should receive a
            float as input and return a float (the resulting activation) as
            output.
        hidden_nodes_activation (Callable[[float], float]): Activation function
            to be used by the hidden nodes of the networks. It should receive a
            float as input and return a float (the resulting activation) as
            output.
        bias_value (Optional[float]): Constant activation value to be used by
            the bias nodes. If `None`, bias nodes won't be used.
        reset_innovations_period (Optional[int]): If None, the innovation IDs of
            the new genes will never be reset. If an int, the innovation IDs
            will be reset with a period (number of generations passed) equal to
            the value specified. As long as the id handler isn't reset, a hidden
            node can't be inserted more than once in a connection between two
            given nodes.
        allow_self_connections (bool): Whether to allow connections connecting a
            node to itself being formed. If a node is connected to itself, it
            considers its last output when calculating its new output.
        disable_inherited_connection_chance (bool): During a sexual reproduction
            between two genomes, this constant specifies the chance of a
            connection in the newly born genome being disabled if it's disabled
            on at least one of the parent genomes.
        initial_node_activation (float): Initial activation value cached by a
            node when it's created or reset.
    """

    def __init__(self,
                 file_pathname=None,
                 # genome creation
                 out_nodes_activation=nevopy.activations.steepened_sigmoid,
                 hidden_nodes_activation=nevopy.activations.steepened_sigmoid,
                 bias_value=1,
                 # reproduction
                 weak_genomes_removal_pc=0.8,
                 weight_mutation_chance=0.8,
                 new_node_mutation_chance=0.03,
                 new_connection_mutation_chance=0.03,
                 enable_connection_mutation_chance=0.03,
                 disable_inherited_connection_chance=0.75,
                 mating_chance=0.6,
                 interspecies_mating_chance=0.01,
                 rank_prob_dist_coefficient=2,
                 # weight mutation
                 weight_perturbation_pc=0.1,
                 weight_reset_chance=0.1,
                 new_weight_interval=(-1, 1),
                 # genome distance coefficients
                 excess_genes_coefficient=1,
                 disjoint_genes_coefficient=1,
                 weight_difference_coefficient=0.4,
                 # speciation
                 species_distance_threshold=1,
                 species_elitism_threshold=5,
                 # others
                 reset_innovations_period=20,
                 allow_self_connections=True,
                 initial_node_activation=0,):
        values = locals()
        values.pop("self")
        values.pop("file_pathname")

        if file_pathname is not None:
            raise NotImplemented()  # todo: implementation

        self.__dict__.update(values)

    def __getitem__(self, key):
        return self.__dict__[key]
