"""
TODO
"""


class IdHandler:
    """ Handles the assignment of IDs to new nodes and connections among different genomes.

    todo

    "A possible problem is that the same structural innovation will receive different innovation numbers in the same
    generation if it occurs by chance more than once. However, by keeping a list of the innovations that occurred in the
    current generation, it is possible to ensure that when the same structure arises more than once through independent
    mutations in the same generation, each identical mutation is assigned the same innovation number. Thus, there is no
    resultant explosion of innovation numbers." - Stanley, K. O. & Miikkulainen, R. (2002)
    """

    def __init__(self, num_inputs, num_outputs, has_bias):
        """ todo

        """
        self._node_counter = num_inputs + num_outputs + 1 if has_bias else 0
        self._connection_counter = num_inputs * num_outputs
        self._genome_counter = 0
        self._species_counter = 0
        self._new_connections_ids = {}
        self._new_nodes_ids = {}
        self.reset_counter = 0

    def reset(self):
        """ Resets the cache of new nodes and connections. Should be called at the start of a new generation. """
        # todo: should this rlly be reset?
        # maybe set this according to a config parameter
        self._new_connections_ids = {}
        self._new_nodes_ids = {}
        self.reset_counter = 0

    def next_genome_id(self):
        """ Returns an unique ID for a genome.

        .. warning::
            This method is not safe for parallel calls!
        """
        gid = self._genome_counter
        self._genome_counter += 1
        return gid

    def next_species_id(self):
        """ Returns an unique ID for a species.

        .. warning::
            This method is not safe for parallel calls!
        """
        sid = self._species_counter
        self._species_counter += 1
        return sid

    def cached_hids(self):
        return self._new_nodes_ids

    def get_hidden_node_id(self, node):
        """ TODO
        """
        src_id, dest_id = node.parent_connection_nodes[0].id, node.parent_connection_nodes[1].id
        if src_id is None or dest_id is None:
            raise RuntimeError("Trying to generate an ID to a node whose parents (one or both) have \"None\" IDs!")

        if src_id in self._new_nodes_ids:
            if dest_id in self._new_nodes_ids[src_id]:
                return self._new_nodes_ids[src_id][dest_id]
        else:
            self._new_nodes_ids[src_id] = {}

        hid = self._node_counter
        self._node_counter += 1
        self._new_nodes_ids[src_id][dest_id] = hid
        return hid

    def get_connection_id(self, connection):
        """ TODO
        """
        src_id, dest_id = connection.from_node.id, connection.to_node.id
        if src_id in self._new_connections_ids:
            if dest_id in self._new_connections_ids[src_id]:
                return self._new_connections_ids[src_id][dest_id]
        else:
            self._new_connections_ids[src_id] = {}

        cid = self._connection_counter
        self._connection_counter += 1
        self._new_connections_ids[src_id][dest_id] = cid
        return cid
