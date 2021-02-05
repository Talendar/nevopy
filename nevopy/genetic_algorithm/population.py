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

""" Implements a generalizable genetic algorithm that can be used by different
neuroevolution algorithms.
"""

import logging
import random
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from nevopy import processing
from nevopy.base_genome import BaseGenome
from nevopy.base_population import BasePopulation
from nevopy.callbacks import Callback
from nevopy.callbacks import CompleteStdOutLogger
from nevopy.callbacks import History
from nevopy.callbacks import SimpleStdOutLogger
from nevopy.genetic_algorithm.config import GeneticAlgorithmConfig
from nevopy.utils import utils

_logger = logging.getLogger(__name__)


class GeneticPopulation(BasePopulation):
    """ Implementation of a generalizable genetic algorithm.

    This class implements a generalizable genetic algorithm that can be used by
    different neuroevolution algorithms. The algorithm is used to evolve a
    population of genomes (instances of a subclass of :class:`.BaseGenome`).

    This class does not make strong assumptions about the type of genome it is
    dealing with, so it does not take into account the type of encoding the
    genome uses or how it processes input. This allows the implemented algorithm
    to be used in a wide range of scenarios.

    The implemented genetic algorithm uses (optionally) a speciation scheme
    similar to the one used by the NEAT algorithm :cite:`stanley:ec02`. The
    computation of the distance between the genomes, however, is not implemented
    here, but on the subclass that implements class:`.BaseGenome`.

    When subclassing this class, you probably won't need to override the
    :meth:`.GeneticPopulation.evolve` method, which contains the main loop of
    the genetic algorithm.

    To better understand the default behaviour of the algorithm, it's
    recommended to read the docs of the methods :meth:`.speciate` and
    :meth:`.reproduction`.

    Example:

        Example using :class:`.FixedTopologyGenome` as the base genome type:

        .. code-block:: python

            def fitness_func(genome):
                \"\"\"
                Function that takes a genome as input and returns the genome's
                fitness (a float) as output.
                \"\"\"
                # ...


            # Genome that's gonna serve as a model for your population:
            base_genome = FixedTopologyGenome(
                layers=[TFDenseLayer(32, activation="relu"),
                        TFDenseLayer(1, activation="sigmoid")],
                input_shape=my_input_shape,  # shape of your input samples
            )

            # Creating and evolving a population:
            population = GeneticPopulation(size=100,
                                           base_genome=base_genome)
            history = population.evolve(generations=100,
                                        fitness_function=fitness_func)

            # Visualizing the evolution of the population's fitness:
            history.visualize()

            # Retrieving and visualizing the fittest genome of the population:
            best_genome = population.fittest()
            best_genome.visualize()

    Args:
        size (int): Number of genomes (constant) in the population.
        base_genome (BaseGenome): Instance of a subclass of :class:`.BaseGenome`
            that will serve as a model/base for all the population's genomes.
        config (Optional[GeneticConfig]): The settings of the evolutionary
            process. If `None`, the default settings will be used.
        processing_scheduler (Optional[ProcessingScheduler]): Processing
            scheduler to be used by the population. If `None`, a new instance of
            :class:`RayProcessingScheduler` will be used as scheduler.
        speciation (bool): Whether the genetic algorithm used to evolve the
            genomes should use speciation or not.
    """

    #: Default processing scheduler used by instances of this class.
    DEFAULT_SCHEDULER = processing.RayProcessingScheduler

    def __init__(self,
                 size: int,
                 base_genome: BaseGenome,
                 config: Optional[GeneticAlgorithmConfig] = None,
                 processing_scheduler: Optional[
                     processing.ProcessingScheduler] = None,
                 speciation: bool = False,
    ) -> None:
        super().__init__(size=size,
                         processing_scheduler=(
                             processing_scheduler
                             if processing_scheduler is not None
                             else GeneticPopulation.DEFAULT_SCHEDULER())
                         )

        # Base genome:
        self._base_genome = base_genome
        if self._base_genome.input_shape is None:
            raise ValueError("The base genome's input shape has not been "
                             "defined! Pass an input shape to the genome's "
                             "constructor or feed a sample input to the "
                             "genome.")

        # Config:
        self._config = (config if config is not None
                        else GeneticAlgorithmConfig())

        if (self._base_genome.config is not None
                and self._base_genome.config != self._config):
            raise ValueError("The base genome was assigned a different config "
                             "object than the one used by the population!")

        self._base_genome.config = self._config

        # Utility instance variables:
        self._mass_extinction_counter = 0
        self._past_best_fitness = None  # type: Optional[float]
        self._last_improvement = 0

        self._cached_rank_prob_dist = utils.rank_prob_dist(
            size=self._size,
            coefficient=self._config.rank_prob_dist_coefficient,
        )

        # Initial genomes:
        self.genomes = [self._base_genome.random_copy()
                        for _ in range(self._size)]

        # Speciation:
        self._speciation = speciation
        self.species = [DefaultSpecies(creation_gen=0,
                                       members=self.genomes[:])]
        self.species[0].update_representative()

    @property
    def config(self):
        return self._config

    def evolve(self,
               generations: int,
               fitness_function: Callable[[BaseGenome], float],
               callbacks: Optional[List[Callback]] = None,
               verbose: int = 2,
               **kwargs,  # pylint: disable=unused-argument
    ) -> History:
        """ Evolves the population using a genetic algorithm.

        Main method of this class. It contains the main loop of the genetic
        algorithm used to evolve the population of genomes.

        Args:
            generations (int): Number of generations for the algorithm to run. A
                generation is completed when all the population's genomes have
                been processed and reproduction and speciation have occurred.
            fitness_function (Callable[[BaseGenome], float]): Fitness function
                to be used to evaluate the fitness of individual genomes. It
                must receive a genome as input and produce a float (the genome's
                fitness) as output.
            callbacks (Optional[List[Callback]]): List with instances of
                :class:`.Callback` that will be called during the evolutionary
                session. By default, a :class:`.History` callback is always
                included in the list. A :class:`.CompleteStdOutLogger` or a
                :class:`.SimpleStdOutLogger` might also be included, depending
                on the value passed to the ``verbose`` param.
            verbose (int): Verbose level (logging on stdout). Options: 0 (no
                verbose), 1 (light verbose) and 2 (heavy verbose).

        Returns:
            A :class:`.History` object containing useful information recorded
            during the evolutionary process.
        """
        # Preparing callbacks:
        if callbacks is None:
            callbacks = []

        history_callback = History()
        callbacks.append(history_callback)

        if verbose >= 2:
            callbacks.append(CompleteStdOutLogger())
        elif verbose == 1:
            callbacks.append(SimpleStdOutLogger())

        for cb in callbacks:
            cb.population = self

        # Resetting improvement records:
        self._last_improvement = 0
        self._past_best_fitness = float("-inf")

        # Resetting mass extinction counter:
        self._mass_extinction_counter = 0
        self._config.update_mass_extinction(0)

        ############################### Evolving ###############################
        self.stop_evolving = False
        generation_num = 0
        for generation_num in range(generations):
            # CALLBACK: on_generation_start
            for cb in callbacks:
                cb.on_generation_start(generation_num, generations)

            # Calculating and assigning FITNESS:
            fitness_results = self.scheduler.run(
                items=self.genomes,
                func=fitness_function
            )  # type: Sequence[float]

            for genome, fitness in zip(self.genomes, fitness_results):
                genome.fitness = fitness

            best = self.fittest()

            # CALLBACK: on_fitness_calculated
            avg_fitness = self.average_fitness()
            for cb in callbacks:
                cb.on_fitness_calculated(best_fitness=best.fitness,
                                         avg_fitness=avg_fitness)

            # Checking if fitness improved:
            improv_diff = best.fitness - self._past_best_fitness
            improv_min_pc = self._config.maex_improvement_threshold_pc

            if improv_diff >= abs(self._past_best_fitness * improv_min_pc):
                self._mass_extinction_counter = 0
                self._past_best_fitness = best.fitness
            else:
                self._mass_extinction_counter += 1

            self._config.update_mass_extinction(self._mass_extinction_counter)

            # CALLBACK: on_mass_extinction_counter_updated
            for cb in callbacks:
                cb.on_mass_extinction_counter_updated(
                    self._mass_extinction_counter
                )

            preys = 0

            # MASS EXTINCTION:
            if (self._mass_extinction_counter
                    >= self._config.mass_extinction_threshold):
                # CALLBACK: on_mass_extinction_start
                for cb in callbacks:
                    cb.on_mass_extinction_start()

                self.mass_extinction(best_genome=best)
                assert len(self.genomes) == self.size, ("The number of genomes "
                                                        "doesn't match the "
                                                        "population's size!")
            # REPRODUCTION:
            else:
                # CALLBACK: on_reproduction_start
                for cb in callbacks:
                    cb.on_reproduction_start()

                preys = self.reproduction()

            # SPECIATION
            if self._speciation:
                # CALLBACK: on_speciation_start
                for cb in callbacks:
                    cb.on_speciation_start()

                self.speciate(current_generation=generation_num)
            else:
                assert len(self.species) == 1, (
                    "Invalid number of species {len(self.species)} in a "
                    "population with speciation disabled!"
                )
                self.species[0].members = self.genomes[:]

            # CALLBACK: on_generation_end
            for cb in callbacks:
                cb.on_generation_end(generation_num, generations,
                                     preys=preys)

            # Checking for early stopping:
            if self.stop_evolving:
                break

        ########################################################################

        # CALLBACK: on_evolution_end
        for cb in callbacks:
            cb.on_evolution_end(generation_num)

        return history_callback

    def mass_extinction(self, best_genome: BaseGenome) -> None:
        """ All the genomes in the population (except for the best genome) are
        replaced by new random genomes (random copies of the population's base
        genome).
        """
        self._mass_extinction_counter = 0
        self.genomes = [best_genome] + [self._base_genome.random_copy()
                                        for _ in range(self._size - 1)]

    @staticmethod
    def generate_offspring(args: Tuple[BaseGenome,
                                       Optional[BaseGenome], bool],
    ) -> BaseGenome:
        """ Given one or two genomes (parents), generates a new genome.

        Args:
            args (Tuple[BaseGenome, Optional[BaseGenome], bool]): Tuple
                containing a genome in its first index, another genome or `None`
                in its second index and a bool in its third index. The bool
                indicates whether predatism will occur or not. If it's `True`,
                then the new genome will be randomly generated. If the second
                index is another genome, then the new genome will be generated
                by mating the two given genomes (sexual reproduction). If its
                `None`, the new genome will be a mutated copy (asexual
                reproduction / binary fission) of the genome in the first index.

        Returns:
            A new genome.
        """
        p1, p2, predate = args

        # Predatism:
        if predate:
            return p1.random_copy()

        # Mating (sexual) vs Binary fission (asexual):
        baby = p1.mate(p2) if p2 is not None else p1.deep_copy()

        # Mutation:
        if p2 is None or utils.chance(baby.config.mutation_chance):
            baby.mutate_weights()

        return baby

    def _count_species_offspring(self, num_offspring: int) -> List[int]:
        """ Assigns a number of offspring to each species.

        The number of offspring assigned to each species is proportional to the
        average fitness of the species. This selection method is called
        `roulette wheel selection`.

        Args:
            num_offspring (int): Number of genomes to be generated by all the
                species combined.

        Returns:
            A list with the number of offspring assigned to each species. The
            ordering of the list follows the order of ``self.species``.
        """
        return utils.round_proportional_distribution(
            to_distribute=num_offspring,
            values=[sp.avg_fitness() for sp in self.species],
        )

    def _elitism(self) -> List[BaseGenome]:
        """ Applies elitism and reverse elitism to the population.

        The best genomes are preserved and passed unchanged to the next
        generation (elitism). The weakest genomes are removed and won't be able
        to reproduce (reverse elitism).

        Returns:
            A list with the preserved genomes.
        """
        preserved = []
        for sp in self.species:
            sp.members.sort(key=lambda genome: genome.fitness,
                            reverse=True)
            # DEBUG:
            _logger.debug(
                f"[ELITISM] Sorted genomes ({len(sp.members)}): "
                f"{[g.fitness for g in sp.members]}")

            # Preserving the fittest (elitism)
            if (len(sp.members) >= self._config.species_elitism_threshold
                    and self._config.elitism_pc > 0):
                idx = min(1, int(self._config.elitism_pc * len(sp.members)))
                preserved += sp.members[:idx]

            # Removing the weakest (reverse elitism)
            rmv_pc = self._config.weak_genomes_removal_pc
            rmv_count = int(rmv_pc * len(sp.members))
            if rmv_count > 0:
                sp.members = sp.members[:-rmv_count]

            # DEBUG:
            _logger.debug(
                f"[ELITISM] Sorted genomes after reverse elitism "
                f"({len(sp.members)}): "
                f"{[g.fitness for g in sp.members]}")

        self.genomes = []
        list(map(self.genomes.extend, [sp.members for sp in self.species]))

        return preserved

    def _select_mating_partners(self,
                                offspring_count: int,
    ) -> Tuple[List[BaseGenome], List[Optional[BaseGenome]]]:
        """ Selects the genomes that will reproduce. """
        parents1, parents2 = [], []
        for sp, os_count in zip(self.species,
                                self._count_species_offspring(offspring_count)):
            if os_count == 0:
                continue

            prob_dist = self._cached_rank_prob_dist[:len(sp.members)]
            prob_dist = prob_dist / prob_dist.sum()

            # First parents:
            parents1 += np.random.choice(sp.members,
                                         size=os_count,
                                         p=prob_dist).tolist()

            # Second parents:
            mating_chance = self._config.mating_chance
            interspecies_chance = self._config.interspecies_mating_chance
            non_sp_genomes = [g for g in self.genomes if g not in sp.members]

            for _ in range(os_count):
                if utils.chance(mating_chance):
                    if (len(non_sp_genomes) > 0
                            and utils.chance(interspecies_chance)):
                        # Sexual reproduction (interspecies)
                        parents2.append(random.choice(non_sp_genomes))
                    else:
                        # Sexual reproduction (same species)
                        parents2.append(random.choice(sp.members))
                else:
                    # Asexual reproduction
                    parents2.append(None)

        assert len(parents1) == len(parents2) == offspring_count
        return parents1, parents2

    def _select_prey(self,
                     offspring_count: int) -> np.ndarray:
        """ Selects which genomes will be preys (replaced by a new random
        genome).
        """
        predatism_chance = self._config.predatism_chance
        return np.random.choice([True, False],
                                size=offspring_count,
                                p=[predatism_chance, 1 - predatism_chance])

    def reproduction(self) -> int:
        """ Handles the reproduction of the population's genomes.

        First, the fittest genomes of each species with more than a pre-defined
        number of individuals are selected to be copied unchanged to the next
        generation (elitism). Next, the least fit genomes of each species are
        discarded (reverse elitism). After that, the number of descendants of
        each species is calculated. The number of offspring assigned to each
        species is proportional to the average fitness of the species
        (`roulette wheel selection`). Finally, the reproduction of individuals
        of the same species (and, on rare occasions, between genomes of
        different species as well) occurs.

        Genomes with a higher fitness have a higher chance of leaving offspring.
        Within a species, the chance of a genome reproducing is given by the
        position it occupies in the species fitness rank
        (`rank-based selection`). This means that the reproduction chance of a
        genome is not directly calculated from the genome's fitness, but rather
        from how well positioned is the genome in the fitness rank.

        Some of the behaviour described above follows the original description
        of the NEAT algorithm :cite:`stanley:ec02`.

        Newborn genomes have a chance of being "eaten by a predator", in which
        case they are replaced by new randomly generated genomes. This technique
        is called `predatism`.

        Returns:
            Number of preys (individuals replaced by a random genome).
        """
        new_pop = []  # type: List[BaseGenome]

        # Elitism:
        new_pop += self._elitism()
        offspring_count = self._size - len(new_pop)

        # DEBUG:
        _logger.debug(
            f"[REPRODUCTION] All species genomes "
            f"({len(self.genomes)}): "
            f"{[g.fitness for g in self.genomes]}")

        # Choosing mating partners:
        parents1, parents2 = self._select_mating_partners(offspring_count)

        # DEBUG:
        asexual_count = (np.array(parents2) == None).sum() \
            # pylint: disable=singleton-comparison
        _logger.debug(
            f"[REPRODUCTION] Mating: "
            f"{(offspring_count - asexual_count) / offspring_count:0.2%} | "
            f"Binary fission: {asexual_count / offspring_count:0.2%}"
        )

        # Predatism:
        predate = self._select_prey(offspring_count)

        # DEBUG
        _logger.debug(f"[REPRODUCTION] Preys (predatism): {predate.sum()}")

        # Generating offspring:
        babies = self.scheduler.run(
            items=[(p1, p2, False) if not predate[i] else (p1, None, True)
                   for i, (p1, p2) in enumerate(zip(parents1, parents2))],
            func=GeneticPopulation.generate_offspring,
        )

        new_pop += babies
        self.genomes = new_pop

        assert len(self.genomes) == self.size, ("The number of genomes doesn't "
                                                "match the population's size!")
        return predate.sum()

    def speciate(self, current_generation) -> None:
        """ Divides the population's genomes into species.

        The algorithm follows the speciation scheme of the NEAT algorithm
        :cite:`stanley:ec02`:

        "Each existing species is represented by a random genome inside the
        species from the previous generation. A given genome g in the current
        generation is placed in the first species in which g is compatible with
        the representative genome of that species. This way, species do not
        overlap. If g is not compatible with any existing species, a new species
        is created with g as its representative." - :cite:`stanley:ec02`

        The degree of compatibility between two genomes is given by their
        distance, calculated by the :meth:`.BaseGenome.distance` method. The
        lower the distance the more compatible two genomes are. Two genomes are
        considered compatible if their distance is lower than a pre-defined
        number (:attr:`.GeneticAlgorithmConfig.species_distance_threshold`).

        Species that haven't improved their fitness for a pre-defined number of
        generations are extinct, i.e., they are removed from the population
        and aren't considered for the speciation process.
        """
        extinction_threshold = self._config.species_no_improvement_limit

        # Checking improvements and resetting members:
        surviving_species = []
        for sp in self.species:
            past_best_fitness = sp.best_fitness
            sp.best_fitness = sp.fittest().fitness

            if past_best_fitness is not None:
                if sp.best_fitness > past_best_fitness:
                    # recording improvement generation
                    sp.last_improvement = current_generation

                no_improvement_time = current_generation - sp.last_improvement
                if no_improvement_time <= extinction_threshold:
                    surviving_species.append(sp)

            sp.members = []

        self.species = surviving_species

        # Assigning genomes to species:
        dist_threshold = self._config.species_distance_threshold
        for genome in self.genomes:
            chosen_species = None

            # Checking compatibility with existing species:
            for sp in self.species:
                if sp.compatibility(genome) <= dist_threshold:
                    chosen_species = sp
                    break

            # Creating new species, if necessary:
            if chosen_species is None:
                chosen_species = DefaultSpecies(creation_gen=current_generation)
                chosen_species.representative = genome
                self.species.append(chosen_species)

            # Adding genome to the chosen species:
            chosen_species.members.append(genome)

        # Deleting empty species and updating representatives:
        remaining_species = []
        for sp in self.species:
            if len(sp.members) > 0:
                sp.update_representative()
                remaining_species.append(sp)
        self.species = remaining_species


class DefaultSpecies:
    """ Represents a species.

    In the context of a genetic algorithm, a species is a set of similar (to
    some extent) genomes that can mate in order to generate offspring.

    Args:
        creation_gen (int): Number of the generation in which the species is
            being created.
        members (Optional[List[BaseGenome]]): Initial members of the species.

    Attributes:
        representative (Optional[BaseGenome]): Genome used to represent the
            species.
        members (List[BaseGenome]): List with the genomes that belong to the
            species.
        last_improvement (int): Generation in which the species last showed
            improvement of its fitness. The species fitness in a given
            generation is equal to the fitness of the species fittest genome on
            that generation.
        best_fitness (Optional[float]): The last calculated fitness of the
            species fittest genome.
    """

    def __init__(self,
                 creation_gen: int,
                 members: Optional[List[BaseGenome]] = None) -> None:
        self.representative = None  # type: Optional[BaseGenome]
        self.members = ([] if members is None
                        else members)  # type: List[BaseGenome]

        self._creation_gen = creation_gen
        self.last_improvement = creation_gen
        self.best_fitness = None   # type: Optional[float]

    def update_representative(self) -> None:
        """ Chooses a new representative for the species.

        This implementation follows NEAT, so a random member of the species is
        chosen as its representative.
        """
        self.representative = np.random.choice(self.members)

    def compatibility(self, genome: BaseGenome) -> float:
        """ Returns a float indicating the compatibility of the given genome
        with the species.
        """
        return self.representative.distance(genome)

    def avg_fitness(self) -> float:
        """ Returns the average fitness of the species genomes. """
        return float(np.mean([g.fitness for g in self.members]))

    def fittest(self) -> BaseGenome:
        """ Returns the fittest member of the species. """
        return self.members[int(np.argmax([g.fitness for g in self.members]))]
