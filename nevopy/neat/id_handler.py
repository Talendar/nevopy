"""
TODO
"""

from multiprocessing import Lock


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
        self._genome_counter = 0
        self._species_counter = 0
        self._new_connections_ids = {}
        self._new_nodes_ids = {}
        self._lock = Lock()

    def reset(self):
        """ Resets the cache of new nodes and connections. Should be called at the start of a new generation. """
        # todo: should this rlly be reset?
        self._new_connections_ids = {}
        self._new_nodes_ids = {}

    def genome_id(self):
        """ Returns an unique ID for a genome. """
        with self._lock:
            gid = self._genome_counter
            self._genome_counter += 1
            return gid

    def species_id(self):
        """ Returns an unique ID for a species. """
        with self._lock:
            sid = self._species_counter
            self._species_counter += 1
            return sid

    def hidden_node_id(self, src_id, dest_id):
        """ TODO

        :param src_id:
        :param dest_id:
        :return:
        """
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
        """ TODO

        :param src_id:
        :param dest_id:
        :return:
        """
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
