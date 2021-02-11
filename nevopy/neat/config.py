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

""" This module implements the :class:`NeatConfig` class, used to handle the
settings of the NEAT algorithm.
"""

import nevopy as ne


class NeatConfig(ne.genetic_algorithm.config.GeneticAlgorithmConfig):
    """ Stores the settings of the NEAT algorithm.

    Individual configurations can be ignored (default values will be used), set
    in the arguments of this class constructor or written in a file (pathname
    passed as an argument).

    Some parameters/attributes related to mutation chances expects a tuple with
    two floats, indicating the minimum and the maximum chance of the mutation
    occurring. A value within the given interval is chosen based on the "mass
    extinction factor" (mutation chances increases as the number of consecutive
    generations in which the population has shown no improvement increases). If
    you want a fixed mutation chance, just place the same value on both
    positions of the tuple.

    Args:
        file_pathname (Optional[str]): The pathname of a file from where the
            settings should be loaded.
        **kwargs: Accepts any of the attributes listed for this class. When the
            value of an attribute isn't passed as argument, a default value is
            used. The default values are defined in
            :attr:`.NeatConfig.ATTRIBUTES`.

    Attributes:
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

        weak_genomes_removal_pc (float): Percentage of the least fit individuals
            to be deleted from the population before the reproduction step.
        weight_mutation_chance (Tuple[float, float]): Tuple containing,
            respectively, the minimum and maximum chance of mutating a
            connection gene.
        new_node_mutation_chance (Tuple[float, float]): Tuple containing,
            respectively, the minimum and maximum chance of a new hidden node
            being added to a newly born genome.
        new_connection_mutation_chance (Tuple[float, float]): Tuple containing,
            respectively, the minimum and maximum chance of a new connection
            being added to a newly born genome.
        enable_connection_mutation_chance (Tuple[float, float]): Tuple
            containing, respectively, the minimum and maximum chance of enabling
            a disabled connection in a newly born genome.
        disable_inherited_connection_chance (float): During a sexual
            reproduction between two genomes, this constant specifies the chance
            of a connection in the newly born genome being disabled if it's
            disabled on at least one of the parent genomes.
        mating_chance (float): Chance of a genome reproducing sexually, i.e.,
            by mating / crossing-over with another genome. Decreasing this value
            increases the chance of a genome reproducing asexually, through
            binary fission (copy + mutation).
        interspecies_mating_chance (float): Chance for a sexual reproduction
            (mating / cross-over) to be between genomes of different species.
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

        weight_perturbation_pc (Tuple[float, float]): Tuple containing,
            respectively, the minimum and maximum value for the maximum absolute
            percentage of the perturbation value of the weights. When a
            connection gene is being mutated, it has a chance of having a value
            (the perturbation) added to its weight. This can me summarized as
            follows (`p` is the weight perturbation percentage):
            `current weight <- current weight * (1 + random[-p, p])`.
        weight_reset_chance (Tuple[float, float]): Tuple containing,
            respectively, the minimum and maximum chance of resetting a
            connection's weight during the mutation of a connection gene. The
            reset connection is assigned a new random weight.
        new_weight_interval (Tuple[float, float]): Interval from which the value
            of a new random connection weight will be picked from.

        mass_extinction_threshold (int): If the population's fitness doesn't
            improve for this amount of generations, the whole population, with
            the exception of its most fit genome, will be extinct/deleted and
            replaced by new randomly generated genomes. Here the fitness of a
            population in a given generation is considered to be equal to the
            fitness of the population's most fit genome in that generation. As
            the number of generations without improvements increases, the
            mutations chances (as specified in the settings) also increase. This
            simulates the increase of the evolutionary pressure acting on the
            population.
        maex_improvement_threshold_pc (float): It's considered that the fitness
            of a population improved if, and only if, the population's fitness
            had an increase equivalent to this percentage. As an example,
            suppose that the fitness :math:`f_g` of a population on generation
            :math:`g` is 100 and that this parameter is set to `0.05` (`5%`).
            The fitness :math:`f_{g+1}` of the population in the next generation
            (`g + 1`) is considered to have improved if, and only if,
            :math:`f_{g+1} \\geq 1.05 \\cdot f_g = 105`.

        infanticide_output_nodes (bool): If `True`, newborn genomes with no
            enabled connections incoming to one or more output nodes will be
            deleted and replaced by a new randomly generated genome. Note that
            the term "infanticide" is being used here without any political or
            cultural connotation. It's used because it is the word that best
            describe the phenomenon at hand and is widely used in the scientific
            field of zoology (see
            `this <https://en.wikipedia.org/wiki/Infanticide_(zoology)>`_
            article).
        infanticide_input_nodes (bool): If `True`, newborn genomes with no
            enabled connections leaving one or more input nodes will be deleted
            and replaced by a new randomly generated genome. Note that the term
            "infanticide" is being used here without any political or cultural
            connotation. It's used because it is the word that best describe the
            phenomenon at hand and is widely used in the scientific field of
            zoology (see
            `this <https://en.wikipedia.org/wiki/Infanticide_(zoology)>`_
            article).

        random_genome_bonus_nodes (int): Let `h_bonus` be the argument passed to
            this parameter and `h_max` the maximum number of hidden nodes within
            individuals of the population. When a random genome is created to
            replace one of the population's genomes, the number of hidden nodes
            in it will be a random number picked from the interval
            `[0, h_max + h_bonus]`.
        random_genome_bonus_connections (int): The same as
            :attr:`.NeatConfig.random_genome_max_bonus_hnodes`, except it refers
            to the number of connections involving hidden nodes in the new
            randomly generated genome.

        excess_genes_coefficient (float): Used in the formula that calculates
            the distance between two genomes. It's the :math:`c_1` coefficient
            in :eq:`neat_genome_distance`.
        disjoint_genes_coefficient (float): Used in the formula that calculates
            the distance between two genomes. It's the :math:`c_2` coefficient
            in :eq:`neat_genome_distance`.
        weight_difference_coefficient (float): Used in the formula that
            calculates the distance between two genomes. It's the :math:`c_3`
            coefficient in :eq:`neat_genome_distance`.

        species_distance_threshold (float): Minimum distance, as calculated by
            :eq:`neat_genome_distance`, between two genomes for them to be
            considered as being of the same species. A lower threshold will make
            new species easier to appear, increasing the number of species
            throughout the evolutionary process.
        species_elitism_threshold (int): Species with a number of members
            superior to this threshold will have their fittest member copied
            unchanged to the next generation.
        species_no_improvement_limit (int): If a species doesn't show
            improvement in its best fitness for this amount of generations, it
            will be extinct.

        reset_innovations_period (Optional[int]): If `None`, the innovation IDs
            of the new genes will never be reset. If an `int`, the innovation
            IDs will be reset periodically with a period (number of generations
            passed) equal to the value specified. As long as the id handler
            isn't reset, a hidden node can't be inserted more than once in a
            connection between two given nodes.
        allow_self_connections (bool): Whether to allow or not connections
            connecting a node to itself. If a node is connected to itself, it
            considers its last output when calculating its new output.
        initial_node_activation (float): Initial activation value cached by a
            node when it's created or reset.
    """

    #: Attributes supported by the class and their default values. Each
    #:  attribute can passed as a kwarg in the class' constructor or be
    #:  specified in a config file. Attributes not specified will be initialized
    #:  with a default value.
    ATTRIBUTES = dict(
        # genome creation
        out_nodes_activation=ne.activations.steepened_sigmoid,
        hidden_nodes_activation=ne.activations.steepened_sigmoid,
        bias_value=1,
        # reproduction
        weak_genomes_removal_pc=0.75,
        weight_mutation_chance=(0.7, 0.9),
        new_node_mutation_chance=(0.03, 0.3),
        new_connection_mutation_chance=(0.03, 0.3),
        enable_connection_mutation_chance=(0.03, 0.3),
        disable_inherited_connection_chance=0.75,
        mating_chance=0.7,
        interspecies_mating_chance=0.05,
        rank_prob_dist_coefficient=1.75,
        # weight mutation specifics
        weight_perturbation_pc=(0.1, 0.4),
        weight_reset_chance=(0.1, 0.3),
        new_weight_interval=(-2, 2),
        # mass extinction
        mass_extinction_threshold=15,
        maex_improvement_threshold_pc=0.03,
        # infanticide
        infanticide_output_nodes=True,
        infanticide_input_nodes=True,
        # random genomes
        random_genome_bonus_nodes=-2,
        random_genome_bonus_connections=-2,
        # genome distance coefficients
        excess_genes_coefficient=1,
        disjoint_genes_coefficient=1,
        weight_difference_coefficient=0.5,
        # speciation
        species_distance_threshold=2,
        species_elitism_threshold=5,
        species_no_improvement_limit=15,
        # others
        reset_innovations_period=5,
        allow_self_connections=True,
        initial_node_activation=0
    )

    #: Name of the attributes whose values change according to the mass
    #:  extinction counter (type: Tuple[float, float]).
    MAEX_KEYS = {"weight_mutation_chance",
                 "new_node_mutation_chance",
                 "new_connection_mutation_chance",
                 "enable_connection_mutation_chance",
                 "weight_perturbation_pc",
                 "weight_reset_chance"}

    def __init__(self,
                 file_pathname=None,
                 **kwargs) -> None:
        super().__init__(file_pathname=file_pathname, **kwargs)
