"""
NEvoPy
todo
"""

import nevopy.utils
import nevopy.neat
from multiprocessing import Lock
import numpy as np


# todo: move this class to the appropriate module
class IdHandler:
    """ Handles the assignment of IDs to new nodes and connections among different genomes.

    todo

    "A possible problem is that the same structural innovation will receive different innovation numbers in the same
    generation if it occurs by chance more than once. However, by keeping a list of the innovations that occurred in the
    current generation, it is possible to ensure that when the same structure arises more than once through independent
    mutations in the same generation, each identical mutation is assigned the same innovation number. Thus, there is no
    resultant explosion of innovation numbers." - Stanley, K. O. & Miikkulainen, R. (2002)
    """
    def __init__(self, num_initial_nodes):
        """ todo

        :param num_initial_nodes:
        """
        self._node_counter = num_initial_nodes
        self._connection_counter = 0
        self._new_connections_ids = {}
        self._new_nodes_ids = {}
        self._lock = Lock()

    def reset(self):
        """ Resets the cache of new nodes and connections. Should be called at the start of a new generation. """
        self._new_connections_ids = {}
        self._new_nodes_ids = {}

    def hidden_node_id(self, src_id, dest_id):
        """ todo """
        with self._lock:
            if src_id in self._new_nodes_ids:
                if dest_id in self._new_nodes_ids[src_id]:
                    return self._new_nodes_ids[src_id][dest_id]
            else:
                self._new_nodes_ids[src_id] = {}

            hid = self._node_counter
            self._node_counter += 1
            self._new_nodes_ids[src_id][dest_id] = hid
            return hid

    def connection_id(self, src_id, dest_id):
        """ todo """
        with self._lock:
            if src_id in self._new_connections_ids:
                if dest_id in self._new_connections_ids[src_id]:
                    return self._new_connections_ids[src_id][dest_id]
            else:
                self._new_connections_ids[src_id] = {}

            cid = self._connection_counter
            self._connection_counter += 1
            self._new_connections_ids[src_id][dest_id] = cid
            return cid


xor_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
xor_outputs = [0, 1, 1, 0]


def eval_genome(genome, log=False):
    genome.reset_activations()
    error = 0
    for x, y in zip(xor_inputs, xor_outputs):
        h = genome.process(x)[0]
        error += (y - h) ** 2
        if log:
            print(f"IN: {x}  |  OUT: {h}  |  TARGET: {y}")
    if log:
        print(f"\nError: {error}")
    return 1/error


if __name__ == "__main__":
    id_handler = IdHandler(2 + 1 + 1)
    pop_size = 250
    pop = [nevopy.neat.genome.Genome(num_inputs=2, num_outputs=1, id_handler=id_handler) for _ in range(pop_size)]

    for _ in range(200):
        id_handler.reset()

        # eval fitness
        for genome in pop:
            genome.fitness = eval_genome(genome)

        fitness = [g.fitness for g in pop]
        elite = pop[np.argmax(fitness)]
        print(f"Generation: {_}/{200}\n")
        print(f"Best fitness: {elite.fitness}")
        print(f"Mean fitness: {np.mean(fitness)}\n")

        print("Predictions/outputs:")
        eval_genome(elite, log=True)
        print("\n" + "#"*25 + "\n")

        pop = [elite]
        for i in range(pop_size - 1):
            baby = elite.binary_fission()
            if nevopy.utils.chance(0.2):
                baby.add_random_hidden_node()
            if nevopy.utils.chance(0.2):
                baby.add_random_connection()
            if nevopy.utils.chance(0.32):
                dis_c = [c for c in baby._connections if not c.enabled]
                if dis_c:
                    c = np.random.choice([c for c in baby._connections if not c.enabled])
                    c.enabled = True
            pop.append(baby)

    winner = pop[0]
    print(winner.info())
    eval_genome(winner, log=True)
    winner.visualize()


