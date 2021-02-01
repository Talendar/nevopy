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
handle the settings of `NEvoPY's` fixed-topology neuroevolution algorithms.
"""

from typing import Dict


class FixedTopologyConfig:
    """ Stores the settings of `NEvoPY's` fixed-topology NE algorithms.

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
        | > Implement loading settings from a config file.
        | > Specify the config file organization in the docs.

    Args:
        mating_mode (str): How the exchange of genetic material is supposed to
            happen during a sexual reproduction between two genomes. Options:
            "weights_mating" and "exchange_layers".
        elitism_count(int): Specifies the amount of individuals (among the
            fittest) that will be passed unaltered to the next generation.
        predatism_chance(float): Chance of a newborn genome being "predated"
            (excluded), in which case a new random genome takes its place in the
            next generation.
    """

    #: Name of the mutation chance attributes (type: Tuple[float, float])
    #:  related to mass extinction.
    __MAEX_KEYS = {"mutation_chance",
                   "weight_mutation_chance",
                   "weight_perturbation_pc",
                   "weight_reset_chance"}

    def __init__(self,
                 file_pathname=None,
                 # weight mutation
                 mutation_chance=(0.6, 0.9),
                 weight_mutation_chance=(0.5, 1),
                 weight_perturbation_pc=(0.07, 0.5),
                 weight_reset_chance=(0.07, 0.5),
                 new_weight_interval=(-2, 2),
                 # reproduction
                 weak_genomes_removal_pc=0.5,
                 mating_chance=0.75,
                 mating_mode="weights_mating",
                 rank_prob_dist_coefficient=1.75,
                 elitism_count=3,
                 predatism_chance=0.1,
                 # mass extinction
                 mass_extinction_threshold=25,
                 maex_improvement_threshold_pc=0.03) -> None:
        values = locals()
        values.pop("self")
        values.pop("file_pathname")

        if file_pathname is not None:
            raise NotImplemented  # TODO: implementation

        self.__dict__.update(values)

        # mass extinction ("maex")
        self._maex_cache = {}  # type: Dict[str, float]
        self._maex_counter = 0
        self.update_mass_extinction(0)

    @property
    def maex_counter(self):
        return self._maex_counter

    def __getattribute__(self, key):
        if key in FixedTopologyConfig.__MAEX_KEYS:
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
        for k in FixedTopologyConfig.__MAEX_KEYS:
            base_value, max_value = self.__dict__[k]
            unit = (max_value - base_value) / self.mass_extinction_threshold
            self._maex_cache[k] = (base_value + unit * maex_counter)
