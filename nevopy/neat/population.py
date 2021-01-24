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

""" Implements the main mechanisms of the NEAT algorithm.

This is the main module of `NEvoPY's` implementation of the NEAT algorithm. It
implements the :class:`.NeatPopulation` class, which handles the evolution of a
population/community of genomes.
"""

from typing import Optional, List, Sequence, Dict, Callable

import pickle
from pathlib import Path
import numpy as np

from nevopy.neat.genomes import NeatGenome
from nevopy.neat.genes import NodeGene
from nevopy.neat.config import NeatConfig
from nevopy.neat.id_handler import IdHandler
from nevopy.neat.species import Species
from nevopy.neat.callbacks import (Callback, CompleteStdOutLogger,
                                   SimpleStdOutLogger, History)

from nevopy import utils
from nevopy.processing.base_scheduler import ProcessingScheduler
from nevopy.processing.pool_processing import PoolProcessingScheduler


class NeatPopulation:
    """ Population of individuals (genomes) to be evolved by the NEAT algorithm.

    Main class of `NEvoPY's` implementation of the NEAT algorithm. It represents
    a population of individuals (genomes) to be evolved. The correct term, in
    NEAT's case, is actually "community" (group of populations of two or more
    different species) rather than "population" (subset of individuals of one
    species), since NEAT divides its genomes into species. However, to maintain
    consistency with the neuroevolution literature, the population term is used.

    To use NEAT, most users will need to use only this class. It's main method,
    :meth:`.evolve()`, starts the evolutionary process. By providing a
    processing scheduler, the user is able to specify how the computation of the
    fitness of the population's genomes will occur (whether to use serial or
    parallel processing, CPU or GPU, etc).

    By default, a :class:`.PoolProcessingScheduler` is used. It implements
    parallel processing using (by default) all the CPU cores of the machine
    where the program is running. Alternatively, if you want to run the
    evolution process on multiple machines (cluster) you should check out the
    :class:`.RayProcessingScheduler`.

    Example:

        Suppose you have already defined a function called `calc_fitness` that
        takes a genome as input and calculates its fitness. If the networks
        take 10 input values and outputs 3 values, here is how you can proceed
        to create and evolve a population of 100 genomes using the default
        settings and processing scheduler:

        .. code-block:: python

            pop = NeatPopulation(size=100,
                             num_inputs=10,
                             num_outputs=3)

            pop.evolve(generations=100),
                       fitness_function=calc_fitness)

            best_genome = pop.fittest()
            best_genome.visualize()

    Args:
        size (int): Number of genomes in the population (constant value).
        num_inputs (Optional[int]): Number of input nodes in each genome. If
            `None`, the number of inputs will be inferred from the base genome.
        num_outputs (Optional[int]): Number of output nodes in each genome. If
            `None`, the number of outputs will be inferred from the base genome.
        base_genome (Optional[NeatGenome]): Genome that will serve as a base for
            the randomly generated genomes of the population. If `None`, a new
            genome of the class :class:`.NeatGenome` will be used as the base
            genome.
        config (NeatConfig): The settings of the evolutionary process. If `None`
            the default settings will be used.
        processing_scheduler (Optional[ProcessingScheduler]): Processing
            scheduler to be used to compute the fitness of the population's
            genomes. If `None`, the default scheduler will be used
            :class:`.PoolProcessingScheduler`.

    Attributes:
        config (NeatConfig): The settings of the evolutionary process.
        genomes (Sequence[Genome]): List with the population's genomes.
        species (List[Species]): List with the currently alive species in the
            population.
        stop_evolving (bool): At the start of every generation,
            :meth:`.evolve()` checks if this variable is set to `True`. If it
            is, the evolutionary process is stopped. Useful for callbacks.
    """

    #: Default processing scheduler used by instances of this class.
    _DEFAULT_SCHEDULER = PoolProcessingScheduler

    def __init__(self,
                 size: int,
                 num_inputs: Optional[int] = None,
                 num_outputs: Optional[int] = None,
                 base_genome: Optional[NeatGenome] = None,
                 config: Optional[NeatConfig] = None,
                 processing_scheduler: Optional[ProcessingScheduler] = None,
                 ) -> None:
        # assertions and config
        if base_genome is None:
            if None in (num_inputs, num_outputs):
                raise ValueError(
                    "If you don't pass a base genome as argument, you "
                    "must specify a number of inputs and a number of "
                    "outputs!")

            self.config = config if config is not None else NeatConfig()
            self._base_genome = NeatGenome(num_inputs=num_inputs,
                                           num_outputs=num_outputs,
                                           config=self.config)
        else:
            if None not in (num_inputs, num_outputs):
                if (base_genome.num_inputs != num_inputs
                        or base_genome.num_outputs != num_outputs):
                    raise ValueError(
                        "The specified numbers of inputs and outputs "
                        "are not compatible with the given base "
                        "genome! Expected "
                        f"(in: {base_genome.num_inputs}, "
                        f"out: {base_genome.num_outputs}) but got "
                        f"(in: {num_inputs}, out: {num_outputs}).")

            if config is not None and config != base_genome.config:
                raise ValueError("The `NeatConfig` object passed as argument "
                                 "does not match the `NeatConfig` object of "
                                 "the base genome!")

            self.config = base_genome.config
            self._base_genome = base_genome

        # others instance variables
        self._size = size
        self.stop_evolving = False

        self._scheduler = (processing_scheduler
                           if processing_scheduler is not None
                           else NeatPopulation._DEFAULT_SCHEDULER())

        self._id_handler = IdHandler(self._base_genome.num_inputs,
                                     self._base_genome.num_outputs,
                                     has_bias=self.config.bias_value is not None)

        self._rank_prob_dist = None            # type: Optional[np.ndarray]
        self._invalid_genomes_replaced = None  # type: Optional[int]
        self._mass_extinction_counter = 0

        self.__max_hidden_nodes = None         # type: Optional[int]
        self.__max_hidden_connections = None   # type: Optional[int]

        self._past_best_fitness = None         # type: Optional[float]
        self._last_improvement = 0

        # creating initial genomes
        self.genomes = [self._base_genome.random_copy() for _ in range(size)]

        # creating pioneer species
        new_sp = Species(species_id=self._id_handler.next_species_id(),
                         generation=0)
        new_sp.members = self.genomes[:]
        for m in new_sp.members:
            m.species_id = new_sp.id
        new_sp.random_representative()
        self.species = {new_sp.id: new_sp}

    def fittest(self) -> NeatGenome:
        """ Returns the most fit genome in the population. """
        return self.genomes[int(np.argmax([g.fitness for g in self.genomes]))]

    def evolve(self,
               generations: int,
               fitness_function: Callable[[NeatGenome], float],
               callbacks: Optional[List[Callback]] = None,
               verbose: int = 2) -> History:
        """ Evolves the population of genomes using the NEAT algorithm.

        Args:
            generations (int): Number of generations for the algorithm to run. A
                generation is completed when all the population's genomes have
                been processed and reproduction and speciation has occurred.
            fitness_function (Callable[[NeatGenome], float]): Fitness function to
                be used to evaluate the fitness of individual genomes. It must
                receive a genome as input and produce a float (the genome's
                fitness) as output.
            callbacks (Optional[List[Callback]]): List with instances of
                :class:`.Callback` that will be called during the evolutionary
                session. By default, a :class:`.History` callback is always
                included in the list. A :class:`.CompleteStdOutLogger` or a
                :class:`.SimpleStdOutLogger` might also be included, depending
                on the value passed to the `verbose` param.
            verbose (int): Verbose level (logging on stdout). Options: 0 (no
                verbose), 1 (light verbose) and 2 (heavy verbose).

        Returns:
            A :class:`.History` object containing useful information recorded
            during the evolutionary process.
        """
        # preparing callbacks
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

        # caching the rank-selection probability distribution
        self._calc_prob_dist()

        # resetting improvement record
        self._last_improvement = 0
        self._past_best_fitness = float("-inf")

        # evolving
        self.stop_evolving = False
        generation_num = 0
        for generation_num in range(generations):
            # callback: on_generation_start
            for cb in callbacks:
                cb.on_generation_start(generation_num, generations)

            # calculating fitness
            fitness_results = self._scheduler.run(
                items=self.genomes,
                func=fitness_function
            )  # type: Sequence[float]

            # assigning fitness and adjusted fitness
            for genome, fitness in zip(self.genomes, fitness_results):
                genome.fitness = fitness
                sp = self.species[genome.species_id]
                genome.adj_fitness = genome.fitness / len(sp.members)
            best = self.fittest()

            # counting max number of hidden nodes in one genome
            self.__max_hidden_nodes = np.max([len(g.hidden_nodes)
                                              for g in self.genomes])

            # counting max number of hidden connections in one genome
            self.__max_hidden_connections = np.max([
                len([c for c in g.connections
                     if c.enabled and (c.from_node.type == NodeGene.Type.HIDDEN
                                       or c.to_node.type == NodeGene.Type.HIDDEN)
                     ])
                for g in self.genomes
            ])

            # callback: on_fitness_calculated
            for cb in callbacks:
                cb.on_fitness_calculated(best, self.__max_hidden_nodes,
                                         self.__max_hidden_connections)

            # checking improvements
            improv_diff = best.fitness - self._past_best_fitness
            improv_min_pc = self.config.maex_improvement_threshold_pc
            if improv_diff >= abs(self._past_best_fitness * improv_min_pc):
                self._mass_extinction_counter = 0
                self._past_best_fitness = best.fitness
            else:
                self._mass_extinction_counter += 1
            self.config.update_mass_extinction(self._mass_extinction_counter)

            # callback: on_mass_extinction_counter_updated
            for cb in callbacks:
                cb.on_mass_extinction_counter_updated(
                    self._mass_extinction_counter)

            # checking mass extinction
            if (self._mass_extinction_counter
                    >= self.config.mass_extinction_threshold):
                # callback: on_mass_extinction_start
                for cb in callbacks:
                    cb.on_mass_extinction_start()

                # mass extinction
                self._mass_extinction_counter = 0
                self.genomes = [best] + [self._random_genome_with_extras()
                                         for _ in range(self._size - 1)]
                assert len(self.genomes) == self._size
            else:
                # callback: on_reproduction_start
                for cb in callbacks:
                    cb.on_reproduction_start()

                # reproduction
                self.reproduction()

            # callback: on_speciation_start
            for cb in callbacks:
                cb.on_speciation_start(self._invalid_genomes_replaced)

            # speciation
            self.speciation(generation=generation_num)

            # callback: on_generation_end
            for cb in callbacks:
                cb.on_generation_end(generation_num, generations)

            # early stopping
            if self.stop_evolving:
                break

        # callback: on_evolution_end
        for cb in callbacks:
            cb.on_evolution_end(generation_num)

        return history_callback

    def _random_genome_with_extras(self) -> NeatGenome:
        """ Creates a new random genome with extra hidden nodes and connections.

        The number of hidden nodes in the new genome will be randomly picked
        from the interval `[0, max_hn + bonus_hn]`, where `max_hn` is the
        number of hidden nodes in the genome (of the population) with the
        greatest number of hidden nodes and `bonus_hn` is a bonus value
        specified in the settings. The number of hidden connections in the new
        genome is chosen in a similar way.

        Returns:
            A new random genome with extra hidden nodes and connections.
        """
        new_genome = self._base_genome.random_copy()

        # adding hidden nodes
        max_hnodes = (self.__max_hidden_nodes
                      + self.config.random_genome_bonus_nodes)
        if max_hnodes > 0:
            for _ in range(np.random.randint(low=0, high=(max_hnodes + 1))):
                new_genome.add_random_hidden_node(self._id_handler)

        # adding random connections
        max_hcons = (self.__max_hidden_connections
                     + self.config.random_genome_bonus_connections)
        if max_hcons > 0:
            for _ in range(np.random.randint(low=0, high=(max_hcons + 1))):
                new_genome.add_random_connection(self._id_handler)

        return new_genome

    def generate_offspring(self,
                           species: Species,
                           rank_prob_dist: Sequence) -> NeatGenome:
        """ Generates a new genome from one or more genomes of the species.

        The offspring can be generated either by mating two randomly chosen
        genomes (sexual reproduction) or by cloning a single genome (asexual
        reproduction / binary fission). After the newly born genome is created,
        it has a chance of mutating. The possible mutations are:

            | . Enabling a disabled connection;
            | . Changing the weights of one or more connections;
            | . Creating a new connection between two random nodes;
            | . Creating a new random hidden node.

        Args:
            species (Species): Species from which the offspring will be
                generated.
            rank_prob_dist (Sequence): Sequence (usually a numpy array)
                containing the chances of each of the species genomes being the
                first parent of the newborn genome.

        Returns:
            A newly generated genome.
        """
        g1 = np.random.choice(species.members, p=rank_prob_dist)

        # mating / cross-over
        if utils.chance(self.config.mating_chance):
            # interspecific
            if (len(self.species) > 1
                    and utils.chance(self.config.interspecies_mating_chance)):
                g2 = np.random.choice([g for g in self.genomes
                                       if g.species_id != species.id])
            # intraspecific
            else:
                g2 = np.random.choice(species.members)
            baby = g1.mate(g2)
        # binary_fission
        else:
            baby = g1.deep_copy()

        # enable connection mutation
        if utils.chance(self.config.enable_connection_mutation_chance):
            baby.enable_random_connection()

        # weight mutation
        if utils.chance(self.config.weight_mutation_chance):
            baby.mutate_weights()

        # new connection mutation
        if utils.chance(self.config.new_connection_mutation_chance):
            baby.add_random_connection(self._id_handler)

        # new node mutation
        if utils.chance(self.config.new_node_mutation_chance):
            baby.add_random_hidden_node(self._id_handler)

        # checking genome validity
        valid_out = (not self.config.infanticide_output_nodes
                     or baby.valid_out_nodes())
        valid_in = (not self.config.infanticide_input_nodes
                    or baby.valid_in_nodes())

        # genome is valid
        if valid_out and valid_in:
            return baby

        # invalid genome: replacing with a new random genome
        self._invalid_genomes_replaced += 1
        return self._random_genome_with_extras()

    def _calc_prob_dist(self) -> None:
        """
        Calculates the probability distribution that associates, to each genome
        in a species, the probability of reproducing.
        """
        alpha = self.config.rank_prob_dist_coefficient
        self._rank_prob_dist = np.zeros(len(self.genomes))

        self._rank_prob_dist[0] = 1 - 1 / alpha
        for i in range(1, len(self.genomes)):
            p = self._rank_prob_dist[i - 1] / alpha
            if p < 1e-9:
                break
            self._rank_prob_dist[i] = p

    def reproduction(self) -> None:
        """ Handles the reproduction of the population's genomes

        This method implements the reproduction mechanism described in the
        original paper of the NEAT algorithm :cite:`stanley:ec02`.

        First, the most fit genome of each species with more than a pre-defined
        number of individuals is selected to be passed unchanged to the next
        generation. This is called `elitism`. Next, the least fit genomes of
        each species are discarded. After that, each species is assigned a
        number specifying the number of genomes it will generate for the next
        generation. This number is calculated based on the proportion between
        the total fitness of the population and the adjusted fitness of the
        species (roulette wheel selection). Finally, the reproduction of
        individuals of the same species (and, on rare occasions, between genomes
        of different species as well) occurs.

        Most of the behaviour described above can be adjusted by changing the
        settings of the evolutionary process (see :class:`.NeatConfig`).
        """
        new_pop = []

        # elitism
        for sp in self.species.values():
            sp.members.sort(key=lambda genome: genome.fitness,
                            reverse=True)

            # preserving the most fit individual
            if len(sp.members) >= self.config.species_elitism_threshold:
                new_pop.append(sp.members[0])

            # removing the least fit individuals
            r = int(len(sp.members) * self.config.weak_genomes_removal_pc)
            if 0 < r < len(sp.members):
                r = len(sp.members) - r
                for g in sp.members[r:]:
                    self.genomes.remove(g)
                sp.members = sp.members[:r]

        # todo: disallow members of species that haven't been improving to
        #  reproduce

        # calculating the number of children for each species
        offspring_count = self.offspring_proportion(
            num_offspring=self._size - len(new_pop)
        )

        # creating new genomes
        self._invalid_genomes_replaced = 0
        for sp in self.species.values():
            # reproduction probabilities (rank-based selection)
            prob = self._rank_prob_dist[:len(sp.members)]
            prob_sum = np.sum(prob)

            if abs(prob_sum - 1) > 1e-8:
                # normalizing distribution
                prob = prob / prob_sum

            # generating offspring
            babies = [self.generate_offspring(species=sp,
                                              rank_prob_dist=prob)
                      for _ in range(offspring_count[sp.id])]
            new_pop += babies

        assert len(new_pop) == self._size
        self.genomes = new_pop

        # checking if the innovation ids should be reset
        if (self.config.reset_innovations_period is not None
                and self._id_handler.reset_counter > self.config.reset_innovations_period):
            self._id_handler.reset()
        self._id_handler.reset_counter += 1

    def offspring_proportion(self, num_offspring: int) -> Dict[int, int]:
        """ Calculates the number of descendants each species will leave for the
        next generation.

        Every species is assigned a potentially different number of offspring in
        proportion to the sum of adjusted fitnesses of its member organisms
        :cite:`stanley:ec02`. This is selection method is called `roulette wheel
        selection`.

        Args:
            num_offspring (int): Number of genomes to be generated by all the
                species combined.

        Returns
            A dictionary mapping the ID of each of the population's species to
            the number of descendants it will leave for the next generation.
        """
        adj_fitness = {sp.id: sp.avg_fitness() for sp in self.species.values()}
        total_adj_fitness = np.sum(list(adj_fitness.values()))

        offspring_count = {}
        count = num_offspring
        for sid in self.species:
            offspring_count[sid] = int(num_offspring * adj_fitness[sid] / total_adj_fitness)
            count -= offspring_count[sid]

        for _ in range(count):
            sid = np.random.choice(list(self.species.keys()))
            offspring_count[sid] += 1

        assert np.sum(list(offspring_count.values())) == num_offspring
        return offspring_count

    def speciation(self, generation: int) -> None:
        """ Divides the population's genomes into species.

        The importance of speciation for NEAT:

        "Speciating the population allows organisms to compete primarily within
        their own niches instead of with the population at large. This way,
        topological innovations are protected in a new niche where they have
        time to optimize their structure through competition within the niche.
        The idea is to divide the population into species such that similar
        topologies are in the same species." - :cite:`stanley:ec02`

        The distance (compatibility) between a pair of genomes is calculated
        based on to the number of excess and disjoint genes between them. See
        :meth:`.NeatGenome.distance()` for more information.

        About the speciation process:

        "Each existing species is represented by a random genome inside the
        species from the previous generation. A given genome g in the current
        generation is placed in the first species in which g is compatible with
        the representative genome of that species. This way, species do not
        overlap. If g is not compatible with any existing species, a new species
        is created with g as its representative." - :cite:`stanley:ec02`

        Species that haven't improved their fitness for a pre-defined number of
        generations are extinct, i.e., they are removed from the population
        and aren't considered for the speciation process. This number is
        configurable.

        Args:
            generation (int): Current generation number.
        """
        extinction_threshold = self.config.species_no_improvement_limit

        # checking improvements and resetting members
        removed_sids = []
        for sp in self.species.values():
            past_best_fitness = sp.best_fitness
            sp.best_fitness = sp.fittest().fitness

            if past_best_fitness is not None:
                if sp.best_fitness > past_best_fitness:
                    # updating improvement record
                    sp.last_improvement = generation
                elif (generation - sp.last_improvement) > extinction_threshold:
                    # marking species for extinction (it hasn't shown
                    # improvements in the past few generations)
                    removed_sids.append(sp.id)

            # resetting members
            sp.members = []

        # extinction of unfit species
        for sid in removed_sids:
            self.species.pop(sid)

        # assigning genomes to species
        dist_threshold = self.config.species_distance_threshold
        for genome in self.genomes:
            chosen_species = None

            # checking compatibility with existing species
            for sp in self.species.values():
                if genome.distance(sp.representative) <= dist_threshold:
                    chosen_species = sp
                    break

            # creating a new species, if needed
            if chosen_species is None:
                sid = self._id_handler.next_species_id()
                chosen_species = Species(species_id=sid,
                                         generation=generation)
                chosen_species.representative = genome
                self.species[chosen_species.id] = chosen_species

            # adding genome to species
            chosen_species.members.append(genome)
            genome.species_id = chosen_species.id

        # deleting empty species and updating representatives
        for sp in list(self.species.values()):
            if len(sp.members) == 0:
                self.species.pop(sp.id)
            else:
                sp.random_representative()

    def save(self, abs_path: str) -> None:
        """ Saves the population on the absolute path provided.

        This method uses :py:mod:`pickle` to save the genome. The processing
        scheduler used by the population won't be saved (a new one will have to
        be assigned to the population when it's loaded again).

        Args:
            abs_path (str): Absolute path of the saving file. If the given path
                doesn't end with the suffix ".pkl", it will be automatically
                added to it.
        """
        p = Path(abs_path)
        if not p.suffixes:
            p = Path(str(abs_path) + ".pkl")
        p.parent.mkdir(parents=True, exist_ok=True)

        scheduler_cache = self._scheduler
        self._scheduler = None
        with open(str(p), "wb") as out_file:
            pickle.dump(self, out_file, pickle.HIGHEST_PROTOCOL)

        self._scheduler = scheduler_cache

    @staticmethod
    def load(abs_path: str,
             scheduler: Optional[ProcessingScheduler] = None,
    ) -> "NeatPopulation":
        """ Loads the population from the given absolute path.

        This method uses :py:mod:`pickle` to load the genome.

        Args:
            abs_path (str): Absolute path of the saved ".pkl" file.
            scheduler (Optional[ProcessingScheduler]): Processing scheduler to
                be used by the population. If `None`, the default one will be
                used.

        Returns:
            The loaded population.
        """
        with open(abs_path, "rb") as in_file:
            pop = pickle.load(in_file)

        pop._scheduler = (scheduler if scheduler is not None
                          else NeatPopulation._DEFAULT_SCHEDULER())
        return pop

    def info(self) -> str:
        """
        Returns a string containing relevant information about the population.
        """
        no_hnode = invalid_out = no_cons = 0
        for g in self.genomes:
            invalid_out += 0 if g.valid_out_nodes() else 1
            no_hnode += 1 if len(g.hidden_nodes) == 0 else 0
            no_cons += (0 if [c for c in g.connections
                              if (c.enabled and not c.self_connecting())]
                        else 1)

        return (f"Size: {len(self.genomes)}\n"
                f"Species: {len(self.species)}\n"
                f"Invalid genomes (out nodes): {invalid_out}\n"
                f"No-hidden node genomes: {no_hnode}\n"
                f"No enabled connection (ignore self connections): {no_cons}")
