"""
TODO
"""

import numpy as np


class Species:
    """ Species
    TODO
    """

    def __init__(self, species_id, generation):
        """

        :param species_id:
        :param generation:
        """
        self._id = species_id
        self.representative = None
        self.members = []

        self._creation_gen = generation
        self._last_improvement = generation
        self.fitness_history = []

    @property
    def id(self):
        return self._id

    def random_representative(self):
        """ Randomly chooses a new representative for the species. """
        self.representative = np.random.choice(self.members)

    def avg_fitness(self):
        """ Returns average fitness of the species. """
        return np.mean([g.fitness for g in self.members])

    def fittest(self):
        """ Returns the fittest member of the species. """
        return self.members[int(np.argmax([g.fitness for g in self.members]))]
