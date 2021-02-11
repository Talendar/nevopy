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

""" This module implements the :class:`.FixedTopologyConfig` class, used to
handle the settings of `NEvoPy's` fixed-topology neuroevolutionary algorithms.
"""

from typing import Dict


class GeneticAlgorithmConfig:
    """ Stores the settings to be used by :class:`.GeneticPopulation`.

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

    Todo:
        * Implementation: loading settings from a config file.
        * Specify the config file organization in the docs.

    Args:
        file_pathname (Optional[str]): The pathname of a file from where the
            settings should be loaded.
        **kwargs: Accepts any of the attributes listed for this class. When the
            value of an attribute isn't passed as argument, a default value is
            used. The default values are defined in
            :attr:`.GeneticAlgorithmConfig.ATTRIBUTES`.

    Attributes:
        mutation_chance (Tuple[float, float]): Chance for a mutation to occur
            in a new-born genome.
        weight_mutation_chance (Tuple[float, float]): Chance of each individual
            connection weight of a newborn genome being perturbed during
            mutation.
        weight_perturbation_pc (Tuple[float, float]): Maximum absolute
            percentage of a weight's value that can be added to it during
            mutation. When a connection weight is being mutated, it has a chance
            of being perturbed. This can me summarized as follows (`p` is the
            weight perturbation percentage):
            `current weight <- current weight * (1 + random[-p, p])`.
        weight_reset_chance (Tuple[float, float]): Chance, during mutation, for
            a weight to have its value reset (in which case a new random value
            is assigned to it).
        new_weight_interval (Tuple[float, float]): When a weight is reset, it
            will be assigned with a random value in this interval.

        weak_genomes_removal_pc (float): Percentage of the weakest genomes
            (those with the lowest fitness) to be removed before reproduction
            occurs.
        mating_chance (float): Chance of a genome reproducing sexually, i.e.,
            by mating / crossing-over with another genome. Decreasing this value
            increases the chance of a genome reproducing asexually, through
            binary fission (copy + mutation).
        mating_mode (str): How the exchange of genetic material is supposed to
            happen during a sexual reproduction between two genomes. Options:
            "weights_mating" and "exchange_layers" (the new genome inherits full
            layers from its parents).
        rank_prob_dist_coefficient (float): Coefficient :math:`\\alpha` used to
            calculate the probability distribution used to select genomes for
            reproduction. Basically, the value of this constant can be
            interpreted as follows: the genome with the highest fitness has
            :math:`\\times \\alpha` more chance of being selected for
            reproduction than the second best genome, which, in turn, has
            :math:`\\times \\alpha` more chance of being selected than the third
            best genome, and so forth. This approach to reproduction is called
            rank-based selection.
        predatism_chance (float): Chance of a newborn genome being "predated",
            in which case its replaced by a new randomly generated genome. This
            increases the genetic variability in the population.

        species_distance_threshold (float): Minimum distance between two genomes
            for them to be considered as being of the same species. A lower
            threshold will make new species easier to appear, increasing the
            number of species throughout the evolutionary process.
        species_elitism_threshold (int): Species with a number of members
            superior to this threshold will have one or more of their fittest
            members copied unchanged to the next generation.
        elitism_pc (float): Percentage of the genomes of a big enough species to
            be copied unchanged to the next generation. The number of copied
            genomes is equal to `max(1, ceil(species_size * elitism_pc))`.
        species_no_improvement_limit (int): If a species doesn't show
            improvement in its best fitness for this amount of generations, it
            will be removed from the species' list of the population.

        mass_extinction_threshold (int): If the population's fitness doesn't
            improve for this amount of generations, the whole population, with
            the exception of its fittest genome, will be extinct/deleted and
            replaced by new randomly generated genomes. Here the fitness of a
            population in a given generation is considered to be equal to the
            fitness of its fittest genome in that generation. As the number of
            generations without improvements increases, the mutations chances
            (as specified in the settings) also increase. This simulates the
            increase of the evolutionary pressure acting on the population.
        maex_improvement_threshold_pc (float): It's considered that the fitness
            of a population improved if, and only if, the population's fitness
            had an increase equivalent to this percentage. As an example,
            suppose that the fitness :math:`f_g` of a population on generation
            :math:`g` is 100 and that this parameter is set to `0.05` (`5%`).
            The fitness :math:`f_{g+1}` of the population in the next generation
            (`g + 1`) is considered to have improved if, and only if,
            :math:`f_{g+1} \\geq 1.05 \\cdot f_g = 105`.
    """

    #: Attributes supported by the class and their default values. Each
    #:  attribute can passed as a kwarg in the class' constructor or be
    #:  specified in a config file. Attributes not specified will be initialized
    #:  with a default value.
    ATTRIBUTES = dict(
        # weight mutation
        mutation_chance=(0.6, 0.9),
        weight_mutation_chance=(0.5, 1),
        weight_perturbation_pc=(0.05, 0.5),
        weight_reset_chance=(0.05, 0.5),
        new_weight_interval=(-2, 2),
        # reproduction
        weak_genomes_removal_pc=0.5,
        mating_chance=0.7,
        interspecies_mating_chance=0.05,
        mating_mode="weights_mating",
        rank_prob_dist_coefficient=1.75,
        predatism_chance=0.1,
        # speciation
        species_distance_threshold=1.75,
        species_elitism_threshold=5,
        elitism_pc=0.03,
        species_no_improvement_limit=15,
        # mass extinction
        mass_extinction_threshold=15,
        maex_improvement_threshold_pc=0.03,
    )

    #: Name of the attributes whose values change according to the mass
    #:  extinction counter (type: Tuple[float, float]).
    MAEX_KEYS = {"mutation_chance",
                 "weight_mutation_chance",
                 "weight_perturbation_pc",
                 "weight_reset_chance"}

    def __init__(self,
                 file_pathname=None,
                 **kwargs) -> None:
        values = {}

        # Loading attributes from file:
        if file_pathname is not None:
            raise NotImplementedError("Loading the settings from a file is not "
                                      "supported yet!")  # TODO: implementation

        # Validating kwargs:
        for k in kwargs:
            if k not in type(self).ATTRIBUTES:
                raise ValueError(f"Invalid attribute \"{k}\" passed to the "
                                 "config class constructor kwargs!")

        # Reading kwargs:
        for k in type(self).ATTRIBUTES:
            if k not in values:
                values[k] = (kwargs[k] if k in kwargs
                             else type(self).ATTRIBUTES[k])

        # Updating attributes:
        self.__dict__.update(values)

        # Mass extinction ("maex") related variables:
        self._maex_cache = {}  # type: Dict[str, float]
        self._maex_counter = 0
        self.update_mass_extinction(0)

    @property
    def maex_counter(self) -> int:
        """ Returns the current value stored in the config's mass extinction
        counter.
        """
        return self._maex_counter

    def __getattribute__(self, key):
        """ Overridden in order to provide variable values for the attributes
        related to mass extinction.
        """
        if key in type(self).MAEX_KEYS:
            return self._maex_cache[key]
        return super().__getattribute__(key)

    def update_mass_extinction(self, maex_counter: int) -> None:
        """ Updates the mutation chances based on the current value of the mass
        extinction counter (generations without improvement).

        Args:
            maex_counter (int): Current value of the mass extinction counter
                (generations without improvement).
        """
        self._maex_counter = maex_counter
        for k in type(self).MAEX_KEYS:
            base_value, max_value = self.__dict__[k]
            unit = (max_value - base_value) / self.mass_extinction_threshold
            self._maex_cache[k] = (base_value + unit * maex_counter)
