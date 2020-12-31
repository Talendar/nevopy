"""
NEvoPy
todo
"""

import nevopy.neat
from multiprocessing import Lock
import numpy as np


# todo: move this class to the appropriate module
class IdHandler:
    """ Handles the assignment of IDs to new nodes and connections among different genomes.

    Attributes:
        todo
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


if __name__ == "__main__":
    g = nevopy.neat.genome.Genome(num_inputs=2,
                                  num_outputs=1,
                                  id_handler=IdHandler(5 + 2 + 1))

    """for i in range(16):
        c = np.random.choice([True, False])
        if c:
            con = g.add_random_connection()
            print(f"New connection: {[n.id for n in con]}")
        else:
            print(f"New hidden node: {g.add_hidden_node().id}")
        g.visualize()

    print("\n====================================\n")
    print(f"\nOUTPUT: {g.process(np.array([0, 1, 0, 1, 1]))}\n")"""

    print(g.info())
    for _ in range(20):
        print("\n====================================\n")
        g.mutate_weights()
        print(g.info())
