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

"""
TODO
"""

import numpy as np
from nevopy.neat.genome import Genome, mate_genomes
from nevopy.neat.config import Config
from nevopy.neat.id_handler import IdHandler
from nevopy.neat.species import Species
from nevopy import utils
from nevopy.processing.pool_processing import PoolProcessingScheduler
from typing import Optional, Callable


class Population:
    """
    TODO
    """

    def __init__(self,
                 size,
                 num_inputs,
                 num_outputs,
                 config=None,
                 processing_scheduler=None,
                 reproduction_uses_scheduler=False) -> None:
        self._size = size
        self._num_inputs = num_inputs
        self._num_outputs = num_outputs
        self._rep_uses_scheduler = reproduction_uses_scheduler

        self._scheduler = (PoolProcessingScheduler
                           if processing_scheduler is not None
                           else PoolProcessingScheduler())

        self.config = config if config is not None else Config()
        self._id_handler = IdHandler(num_inputs, num_outputs,
                                     has_bias=self.config.bias_value is not None)
        self._rank_prob_dist = None

        # creating initial genomes
        self.genomes = [Genome(num_inputs=num_inputs,
                               num_outputs=num_outputs,
                               genome_id=self._id_handler.next_genome_id(),
                               config=self.config)
                        for _ in range(size)]

        # creating pioneer species
        new_sp = Species(species_id=self._id_handler.next_species_id(),
                         generation=0)
        new_sp.members = self.genomes[:]
        for m in new_sp.members:
            m.species_id = new_sp.id
        new_sp.random_representative()
        self._species = {new_sp.id: new_sp}

    def fittest(self):
        """ TODO """
        return self.genomes[int(np.argmax([g.fitness for g in self.genomes]))]

    def _set_nodes_id(self, node_list):
        node_queue = [(node, 0) for node in node_list if node.is_id_temp()]
        while node_queue:
            node, tries = node_queue.pop(0)
            parents = node.parent_connection_nodes
            if parents is None:
                raise RuntimeError("Tried to assign an ID to a new node that "
                                   "has parents = \"None\"!")

            if not parents[0].is_id_temp() and not parents[1].is_id_temp():
                node.id = self._id_handler.get_hidden_node_id(node)
            elif tries > 1:
                raise RuntimeError("Couldn't assign an ID to a new added node!")
            else:
                tries += 1
                node_queue.append((node, tries))

    def _set_connections_id(self, connection_list):
        for connection in connection_list:
            if (connection.from_node.is_id_temp()
                    or connection.to_node.is_id_temp()):
                raise RuntimeError("Tried to assign an ID to a connection "
                                   "between nodes with temp IDs! Did you "
                                   "update the nodes IDs before trying to "
                                   "update the connections IDs?")
            connection.id = self._id_handler.get_connection_id(connection)

    def update_ids(self):
        # checking if the innovation ids should be reset
        if (self.config.reset_innovations_period is not None
                and self._id_handler.reset_counter > self.config.reset_innovations_period):
            self._id_handler.reset()
            for genome in self.genomes:
                genome.reset_connections_ids_cache()
        self._id_handler.reset_counter += 1

        # checking genomes
        genomes_ids = set([g.id for g in self.genomes])
        new_nodes, new_connections = [], []
        for genome in self.genomes:
            # assigning ids to new genomes
            if genome.id is None:
                new_id = self._id_handler.next_genome_id()
                if new_id in genomes_ids:
                    raise RuntimeError("The ID handler assigned an existing ID "
                                       "to a genome.")
                genome.id = new_id

            # retrieving new genes
            new_nodes += genome.new_nodes
            new_connections += genome.new_connections

        self._set_nodes_id(new_nodes)
        self._set_connections_id(new_connections)

        # todo: check for duplicated connections

    def evolve(self, generations, fitness_function):
        """

        TODO: remove genomes without enabled connections to the output nodes?

        :param generations:
        :param fitness_function:
        :return:
        """
        # caching the rank-selection probability distribution
        self._calc_prob_dist()

        # evolving
        for generation_num in range(generations):
            # resetting genomes
            for genome in self.genomes:
                genome.reset_news_cache()

            print(f"[{100*(generation_num + 1) / generations :.2f}%] "
                  f"Generation {generation_num+1} of {generations}.\n"
                  f"Number of species: {len(self._species)}")

            # calculating fitness
            print("Calculating fitness... ", end="")
            fitness_results = self._scheduler.run(items=self.genomes,
                                                  func=fitness_function)

            # assigning fitness and adjusted fitness
            for genome, fitness in zip(self.genomes, fitness_results):
                genome.fitness = fitness
                sp = self._species[genome.species_id]
                genome.adj_fitness = genome.fitness / len(sp.members)
            print("done!")

            # info
            best = self.genomes[int(np.argmax([g.fitness
                                               for g in self.genomes]))]
            print(f"Best fitness: {best.fitness}")
            print("Avg. population fitness: "
                  f"{np.mean([g.fitness for g in self.genomes])}")

            # reproduction and speciation
            print("Reproduction... ", end="")
            self.reproduction()
            print("done!\nSpeciation... ", end="")
            self.speciation(generation=generation_num)
            print("done!\n\n" + "#" * 30 + "\n")

    def _generate_offspring(self,
                            species: Species,
                            rank_prob_dist: np.array) -> Genome:
        """ TODO

        Args:
            species:
            rank_prob_dist:

        Returns:

        """
        g1 = np.random.choice(species.members, p=rank_prob_dist)

        # mating / cross-over
        if utils.chance(self.config.mating_chance):
            # interspecific
            if (len(self._species) > 1
                    and utils.chance(self.config.interspecies_mating_chance)):
                g2 = np.random.choice([g for g in self.genomes
                                       if g.species_id != species.id])
            # intraspecific
            else:
                g2 = np.random.choice(species.members)
            baby = mate_genomes(g1, g2)
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
            baby.add_random_connection()

        # new node mutation
        if utils.chance(self.config.new_node_mutation_chance):
            baby.add_random_hidden_node(cached_hids=self._id_handler.cached_hids())

        return baby

    def _calc_prob_dist(self):
        """
        TODO
        """
        alpha = self.config.rank_prob_dist_coefficient
        self._rank_prob_dist = np.zeros(len(self.genomes))

        self._rank_prob_dist[0] = 1 - 1 / alpha
        for i in range(1, len(self.genomes)):
            p = self._rank_prob_dist[i - 1] / alpha
            if p < 1e-9:
                break
            self._rank_prob_dist[i] = p

    def reproduction(self):
        """

        :return:
        """
        new_pop = []

        # elitism
        for sp in self._species.values():
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
        offspring_count = self._offspring_proportion(
            num_offspring=self._size - len(new_pop)
        )

        # creating new genomes
        for sp in self._species.values():
            # reproduction probabilities (rank-based selection)
            prob = self._rank_prob_dist[:len(sp.members)]
            prob_sum = np.sum(prob)

            if abs(prob_sum - 1) > 1e-8:
                # normalizing distribution
                prob = prob / prob_sum
            # print(f"\n{prob}")

            # generating offspring
            if self._rep_uses_scheduler:
                babies = self._scheduler.run(
                     items=list(range(offspring_count[sp.id])),
                     func=lambda i: self._generate_offspring(
                         species=sp, rank_prob_dist=prob
                     )
                )
            else:
                babies = [self._generate_offspring(species=sp,
                                                   rank_prob_dist=prob)
                          for _ in range(offspring_count[sp.id])]
            new_pop += babies

            # todo: infanticide (remove individuals with no in connections to
            #  output nodes or no out connections from input nodes)

        # new population
        self.genomes = new_pop
        self.update_ids()

    def _offspring_proportion(self, num_offspring):
        """ Roulette wheel selection. """
        adj_fitness = {sp.id: sp.avg_fitness() for sp in self._species.values()}
        total_adj_fitness = np.sum(list(adj_fitness.values()))

        offspring_count = {}
        count = num_offspring
        for sid in self._species:
            offspring_count[sid] = int(num_offspring * adj_fitness[sid] / total_adj_fitness)
            count -= offspring_count[sid]

        for _ in range(count):
            sid = np.random.choice(list(self._species.keys()))
            offspring_count[sid] += 1

        assert np.sum(list(offspring_count.values())) == num_offspring
        return offspring_count

    def speciation(self, generation):
        """
        "Each existing species is represented by a random genome inside the
        species from the previous generation. A given genome g in the current
        generation is placed in the first species in which g is compatible with
        the representative genome of that species. This way, species do not
        overlap. If g is not compatible with any existing species, a new species
        is created with g as its representative." - Stanley, K. O.
        """
        extinction_threshold = self.config.species_no_improvement_limit

        # checking improvements and resetting members
        removed_sids = []
        for sp in self._species.values():
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
            self._species.pop(sid)

        # assigning genomes to species
        dist_threshold = self.config.species_distance_threshold
        for genome in self.genomes:
            chosen_species = None

            # checking compatibility with existing species
            for sp in self._species.values():
                if genome.distance(sp.representative) <= dist_threshold:
                    chosen_species = sp
                    break

            # creating a new species, if needed
            if chosen_species is None:
                sid = self._id_handler.next_species_id()
                chosen_species = Species(species_id=sid,
                                         generation=generation)
                chosen_species.representative = genome
                self._species[chosen_species.id] = chosen_species

            # adding genome to species
            chosen_species.members.append(genome)
            genome.species_id = chosen_species.id

        # deleting empty species and updating representatives
        for sp in list(self._species.values()):
            if len(sp.members) == 0:
                self._species.pop(sp.id)
            else:
                sp.random_representative()

