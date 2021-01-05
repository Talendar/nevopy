"""
TODO
"""

import nevopy


class Config:
    """

    :ivar weight_mutation_chance: chance of mutating a connection.
    :ivar weight_perturbation_pc=0.1: maximum absolute percentage value for the perturbation of the weights
    :ivar weight_reset_chance=0.1: chance to reset the weight of a connection (assign it a new random value)
    :ivar new_weight_interval: range of values a newly created weight can have

    :ivar excess_genes_coefficient: used in the formula to calculate the distance between two genomes.
    :ivar disjoint_genes_coefficient: used in the formula to calculate the distance between two genomes.
    :ivar weight_difference_coefficient: used in the formula to calculate the distance between two genomes.

    :ivar species_distance_threshold: minimum distance between two genomes for them to be of the same species.
    :ivar species_elitism_threshold: species with a number of members superior to this threshold will have their fittest
    member copied unchanged to the next generation.
    :ivar weak_genomes_removal_pc: percentage of the least fit individuals to be deleted during reproduction.

    :ivar mating_chance: chance for a reproduction to be of the type "mating" / "cross-over".
    :ivar interspecies_mating_chance: chance for a mating / cross-over to be between individuals of different species.

    :ivar new_node_mutation_chance: chance of a new hidden node being added to the offspring.
    :ivar new_connection_mutation_chance: chance of a new connection being added to the offspring.
    :ivar enable_connection_mutation_chance: chance of enabling a disabled connection in the offspring.
    :ivar rank_prob_dist_coefficient;

    :ivar out_nodes_activation:
    :ivar hidden_nodes_activation:
    :ivar bias_value:

    :ivar reset_innovations_period: if None, the innovation IDs of the new genes will never be reset; if an int, the
    innovation IDs will be reset with a period (number of generations passed) equal to the one specified. As long as the
    id handler isn't reset, a hidden node can't be inserted more than once in a connection between two given nodes.
    :ivar allow_self_connections:
    :ivar disable_inherited_connection_chance:
    :ivar initial_node_activation:
    """

    def __init__(self,
                 file=None,
                 # genome creation
                 out_nodes_activation=nevopy.activations.steepened_sigmoid,
                 hidden_nodes_activation=nevopy.activations.steepened_sigmoid,
                 bias_value=1,
                 # reproduction
                 weak_genomes_removal_pc=0.8,
                 weight_mutation_chance=0.8,
                 new_node_mutation_chance=0.05,
                 new_connection_mutation_chance=0.05,
                 enable_connection_mutation_chance=0.05,
                 disable_inherited_connection_chance=0.75,
                 mating_chance=0.5,
                 interspecies_mating_chance=0.01,
                 rank_prob_dist_coefficient=1.7,
                 # weight mutation
                 weight_perturbation_pc=0.1,
                 weight_reset_chance=0.1,
                 new_weight_interval=(-1, 1),
                 # genome distance coefficients
                 excess_genes_coefficient=1,
                 disjoint_genes_coefficient=1,
                 weight_difference_coefficient=0.4,
                 # speciation
                 species_distance_threshold=2,
                 species_elitism_threshold=5,
                 # others
                 reset_innovations_period=25,
                 allow_self_connections=True,
                 initial_node_activation=0,):
        """
        TODO
        """
        values = locals()
        values.pop("self")
        values.pop("file")

        if file is not None:
            pass

        self.__dict__.update(values)

    def __getitem__(self, key):
        return self.__dict__[key]

    def print(self):
        for k, v in self.__dict__.items():
            print(k, v)
