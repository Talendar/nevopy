"""

"""

import numpy as np

from nevopy.neat.genome import *
from nevopy.neat.config import Config
from nevopy.neat.id_handler import IdHandler
from nevopy.neat.species import Species
from nevopy import utils
from nevopy.parallel_processing.scheduler import JobScheduler
from nevopy.parallel_processing.worker import LocalWorker
import multiprocessing


class Population:
    """

    """

    def __init__(self, size, num_inputs, num_outputs, job_scheduler=None, config=None):
        """

        :param size:
        :param num_inputs:
        :param num_outputs:
        :param config:
        """
        #if job_scheduler is None:
            #self._job_scheduler = JobScheduler(worker_list=[LocalWorker(worker_id=0)])

        self._size = size
        self._num_inputs = num_inputs
        self._num_outputs = num_outputs

        self.config = config if config is not None else Config()
        self._id_handler = IdHandler(num_inputs, num_outputs, has_bias=self.config.bias_value is not None)

        # creating initial genomes
        self.genomes = [Genome(num_inputs=num_inputs,
                               num_outputs=num_outputs,
                               genome_id=self._id_handler.next_genome_id(),
                               config=self.config)
                        for _ in range(size)]

        # creating pioneer species
        new_sp = Species(species_id=self._id_handler.next_species_id(), generation=0)
        new_sp.members = self.genomes[:]
        for m in new_sp.members:
            m.species_id = new_sp.id
        new_sp.random_representative()
        self._species = {new_sp.id: new_sp}

    def fittest(self):
        """ TODO """
        return self.genomes[np.argmax([g.fitness for g in self.genomes])]

    def _set_nodes_id(self, node_list):
        node_queue = [(node, 0) for node in node_list if node.is_id_temp()]
        while node_queue:
            node, tries = node_queue.pop(0)
            parents = node.parent_connection_nodes
            if parents is None:
                raise RuntimeError("Tried to assign an ID to a new node that has parents = \"None\"!")

            if not parents[0].is_id_temp() and not parents[1].is_id_temp():
                node.id = self._id_handler.get_hidden_node_id(node)
            elif tries > 1:
                raise RuntimeError("Couldn't assign an ID to a new added node!")
            else:
                tries += 1
                node_queue.append((node, tries))

    def _set_connections_id(self, connection_list):
        for connection in connection_list:
            if connection.from_node.is_id_temp() or connection.to_node.is_id_temp():
                raise RuntimeError("Tried to assign an ID to a connection between nodes with temp IDs! Did you update "
                                   "the nodes IDs before trying to update the connections IDs?")
            connection.id = self._id_handler.get_connection_id(connection)

    def update_ids(self):
        # checking if the innovation ids should be reset
        if self.config.reset_innovations_period is not None \
                and self._id_handler.reset_counter > self.config.reset_innovations_period:
            self._id_handler.reset()
            for genome in self.genomes:
                genome.reset_connections_ids_cache()
            print("\n\n>> RESET IDS << \n")

        self._id_handler.reset_counter += 1

        # checking genomes
        genomes_ids = set([g.id for g in self.genomes])
        new_nodes, new_connections = [], []
        for genome in self.genomes:
            # assigning ids to new genomes
            if genome.id is None:
                new_id = self._id_handler.next_genome_id()
                assert new_id not in genomes_ids, "Error: ID handler assigned an existing ID to a genome."
                genome.id = new_id
            # retrieving new genes
            new_nodes += genome.new_nodes
            new_connections += genome.new_connections

        self._set_nodes_id(new_nodes)
        self._set_connections_id(new_connections)

    def evolve(self, generations, fitness_function):
        """

        :param generations:
        :param fitness_function:
        :return:
        """
        # evolving
        for generation_num in range(generations):
            # resetting genomes
            for genome in self.genomes:
                genome.reset_news_cache()

            print(f"[{100*(generation_num + 1) / generations :.2f}%] Generation {generation_num+1} of {generations}.")
            print(f"Number of species: {len(self._species)}")

            # calculating fitness
            print("Calculating fitness... ", end="")
            #fitness_results = self._job_scheduler.run(items=self.genomes, func=fitness_function)
            fitness_results = [fitness_function(genome) for genome in self.genomes]

            for genome, fitness in zip(self.genomes, fitness_results):
                genome.fitness = fitness
                sp = self._species[genome.species_id]
                genome.adj_fitness = genome.fitness / len(sp.members)
            print("done!")

            # log
            best = self.genomes[np.argmax([g.fitness for g in self.genomes])]
            print(f"Best fitness: {best.fitness}")
            print(f"Avg. population fitness: {np.mean([g.fitness for g in self.genomes])}")

            # reproduction and speciation
            print("Reproduction... ", end="")
            self.reproduction()
            print("done!\nSpeciation... ", end="")
            self.speciation(generation=generation_num)
            print("done!\n\n" + "#" * 30 + "\n")

    def _generate_offspring(self, params):
        sp, rank_prob_dist = params["species"], params["prob_dist"]
        g1 = np.random.choice(sp.members, p=rank_prob_dist)

        # mating / cross-over
        if utils.chance(self.config.mating_chance):
            # interspecific
            if len(self._species) > 1 and utils.chance(self.config.interspecies_mating_chance):
                g2 = np.random.choice([g for g in self.genomes if g.species_id != sp.id])
            # intraspecific
            else:
                g2 = np.random.choice(sp.members)
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

        # adding new genome
        return baby

    def reproduction(self):
        """

        :return:
        """
        new_pop = []

        # elitism
        for sp in self._species.values():
            sp.members.sort(key=lambda g: g.fitness)

            # preserving the most fit individual
            if len(sp.members) >= self.config.species_elitism_threshold:
                new_pop.append(sp.members[-1])

            # removing the least fit individuals
            r = int(len(sp.members) * self.config.weak_genomes_removal_pc)
            if 0 < r < len(sp.members):
                for g in sp.members[:r]:
                    self.genomes.remove(g)
                sp.members = sp.members[r:]

        # new genomes
        offspring_count = self._offspring_proportion(num_offspring=self._size - len(new_pop))
        for sp in self._species.values():
            # assigning reproduction probabilities (rank-based selection)
            rank_prob_dist = np.zeros(len(sp.members))
            for i in reversed(range(len(sp.members))):
                rank_prob_dist[i] = self.config.rank_prob_dist_coefficient ** (i / len(sp.members))
            rank_prob_dist /= np.sum(rank_prob_dist)

            # generating offspring
            # todo: parallelize
            #babies = self._job_scheduler.run(items=[{"species": sp, "prob_dist": rank_prob_dist}] * offspring_count[sp.id],
                                       #func=self._generate_offspring)
            babies = [self._generate_offspring({"species": sp, "prob_dist": rank_prob_dist})
                      for _ in range(offspring_count[sp.id])]
            new_pop += babies

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

        assert np.sum(list(offspring_count.values())) == num_offspring  # todo: delete
        return offspring_count

    def speciation(self, generation):
        """

        "Each existing species is represented by a random genome inside the species from the previous generation. A
        given genome g in the current generation is placed in the first species in which g is compatible with the
        representative genome of that species. This way, species do not overlap. If g is not compatible with any
        existing species, a new species is created with g as its representative."
        - Stanley, K. O. & Miikkulainen, R. (2002)
        """
        # resetting species members
        for sp in self._species.values():
            # todo: check species improvement (extinction)
            sp.members = []

        # assigning genomes to species
        for genome in self.genomes:
            genome_species = None

            # checking compatibility with existing species
            for sp in self._species.values():
                if genome.distance(sp.representative) <= self.config.species_distance_threshold:
                    genome_species = sp
                    break

            # creating a new species, if needed
            if genome_species is None:
                genome_species = Species(species_id=self._id_handler.next_species_id(), generation=generation)
                genome_species.representative = genome
                self._species[genome_species.id] = genome_species

            # adding genome to species
            genome_species.members.append(genome)
            genome.species_id = genome_species.id

        # deleting empty species and updating representatives
        for sp in list(self._species.values()):
            if len(sp.members) == 0:
                self._species.pop(sp.id)
            else:
                sp.random_representative()
