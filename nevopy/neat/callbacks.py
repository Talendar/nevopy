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

""" Defines a base interface for all callbacks and implements simple callbacks.

In the context of :mod:`nevopy.neat`, callbacks are utilities called at certain
points during the evolution of a population. They are a powerful tool to
customize the behavior of an evolutionary session of the NEAT algorithm.

Example:

    To implement your own callback, simply create a class that inherits from
    :class:`.Callback` and pass an instance of it to
    :meth:`.neat.population.Population.evolve()`.

    .. code-block:: python

        class MyCallback(Callback):
            def on_generation_start(self,
                                    current_generation,
                                    total_generations):
                print("This is printed at the start of every generation!")
                print(f"Starting generation {current_generation} of "
                      f"{total_generations}.")

        pop.evolve(generations=100,
                   fitness_function=my_func,
                   callbacks=[MyCallback()])
"""

from typing import Optional, List
import nevopy.neat as neat

import numpy as np
from columnar import columnar
from click import style


class Callback:
    """ Abstract base class used to build new callbacks.

    All callbacks passed as argument to
    :meth:`.neat.population.Population.evolve()` should inherit this class.

    Attributes:
        population (Population): Reference to the instance of
            :class:`.neat.population.Population` being evolved by the NEAT
            algorithm.
    """

    def __init__(self) -> None:
        self.population = None  # type: Optional[neat.population.Population]

    def on_generation_start(self,
                            current_generation: int,
                            total_generations: int) -> None:
        """ Called at the beginning of each new generation.

        Subclasses should override this method for any actions to run.

        Args:
            current_generation (int): Number of the current generation.
            total_generations (int): Total number of generations.
        """

    def on_fitness_calculated(self,
                              best_genome: neat.genome.Genome,
                              max_hidden_nodes: int,
                              max_hidden_connections: int) -> None:
        """ Called right after the fitness of the population's genomes and the
        maximum number of hidden nodes and hidden connections in the population
        are calculated.

        Subclasses should override this method for any actions to run.

        Args:
            best_genome (Genome): The most fit genome of the generation.
            max_hidden_nodes (int): Maximum number of hidden nodes found in a
                genome in the population.
            max_hidden_connections (int): Maximum number of hidden connections
                (connections involving at least one hidden node) found in a
                genome in the population.
        """

    def on_mass_extinction_counter_updated(self, mass_ext_counter: int) -> None:
        """ Called right after the mass extinction counter is updated.

        Subclasses should override this method for any actions to run.

        Args:
            mass_ext_counter (int): Current value of the mass extinction
                counter.
        """

    def on_mass_extinction_start(self) -> None:
        """ Called when at the beginning of a mass extinction event.

        Subclasses should override this method for any actions to run.

        Note:
            When this is called, :meth:`on_reproduction_start()` is not called.
        """

    def on_reproduction_start(self) -> None:
        """ Called at the beginning of the reproductive process.

        Subclasses should override this method for any actions to run.

        Note:
            When this is called, :meth:`on_mass_extinction_start()` is not
            called.
        """

    def on_speciation_start(self, invalid_genomes_replaced: int) -> None:
        """ Called at the beginning of the speciation process.

        Called after the reproduction or mass extinction have occurred and
        immediately before the speciation process.

        Subclasses should override this method for any actions to run.

        Args:
            invalid_genomes_replaced (int): Number of invalid genomes replaced
                during the reproduction step.
        """

    def on_generation_end(self,
                          current_generation: int,
                          total_generations: int) -> None:
        """ Called at the end of each generation.

        Subclasses should override this method for any actions to run.

        Args:
            current_generation (int): Number of the current generation.
            total_generations (int): Total number of generations.
        """


class CompleteStdOutLogger(Callback):
    """ Callback that prints info to the standard output.

    Note:
        This callback is heavily verbose / wordy! Consider using the reduced
        logger (:class:`SimpleStdOutLogger`) if you don't like too much text on
        your screen.
    """

    _SEP_SIZE = 80
    _POS_COLOR = "green"
    _NEG_COLOR = "red"
    _NEUT_COLOR = "white"
    __TAB_HEADER = ["NAME", "CURRENT", "PAST", "INCREASE", "INCREASE (%)"]
    _TAB_ARGS = dict(no_borders=False, justify="c", min_column_width=14)

    def __init__(self, precision=2, colors=True):
        super().__init__()
        self.p = precision
        self.colors = colors
        self._past_num_species = 0
        self._past_best_fitness = 0.0
        self._past_avg_fitness = 0.0
        self._past_max_hidden_nodes = 0
        self._past_max_hidden_connections = 0
        self._past_new_node_mutation = 0.0
        self._past_new_con_mutation = 0.0
        self.__table = None  # type: Optional[List[List[str]]]

    @staticmethod
    def __inc_txt_color(txt, past, current):
        return style(txt,
                     fg=(CompleteStdOutLogger._POS_COLOR if current > past
                         else CompleteStdOutLogger._NEG_COLOR if current < past
                         else CompleteStdOutLogger._NEUT_COLOR))

    def on_generation_start(self,
                            current_generation: int,
                            total_generations: int) -> None:
        g_cur, g_tot = current_generation, total_generations
        sp_cur = len(self.population.species)
        sp_inc = f"{sp_cur - self._past_num_species:+0d}"

        if self.colors:
            sp_inc = self.__inc_txt_color(sp_inc,
                                          self._past_num_species, sp_cur)

        print("\n"
              f"[{(g_cur + 1) / g_tot:.2%}] "
              f"Generation {g_cur + 1} of {g_tot}.\n"
              f"Number of species: {sp_cur} ({sp_inc})")
        print(f"Calculating fitness "
              f"(last: {self._past_best_fitness:.{self.p}E})... ", end="")
        self._past_num_species = sp_cur

    def on_fitness_calculated(self,
                              best_genome: neat.genome.Genome,
                              max_hidden_nodes: int,
                              max_hidden_connections: int) -> None:
        # best fitness
        b_fit = best_genome.fitness
        b_fit_inc = b_fit - self._past_best_fitness
        b_fit_pc = (float("inf") if self._past_best_fitness == 0
                    else b_fit_inc/self._past_best_fitness)

        b_fit_inc = f"{b_fit_inc:+0.{self.p}E}"
        b_fit_pc = f"{b_fit_pc:+0.{self.p}%}"

        # max hidden nodes
        mh_nodes = max_hidden_nodes
        mh_nodes_inc = f"{mh_nodes - self._past_max_hidden_nodes:+0d}"

        # max hidden connections
        mh_cons = max_hidden_connections
        mh_cons_inc = f"{mh_cons - self._past_max_hidden_connections:+0d}"

        # avg pop fitness
        avg_fit = np.mean([g.fitness for g in self.population.genomes])
        avg_fit_inc = avg_fit - self._past_avg_fitness
        avg_fit_pc = (float("inf") if self._past_avg_fitness == 0
                      else avg_fit_inc / self._past_avg_fitness)

        avg_fit_inc = f"{avg_fit_inc:+0.{self.p}E}"
        avg_fit_pc = f"{avg_fit_pc:+0.{self.p}%}"

        # colors
        if self.colors:
            b_fit_inc = self.__inc_txt_color(b_fit_inc, self._past_best_fitness,
                                             b_fit)
            b_fit_pc = self.__inc_txt_color(b_fit_pc, self._past_best_fitness,
                                            b_fit)
            mh_nodes_inc = self.__inc_txt_color(mh_nodes_inc,
                                                self._past_max_hidden_nodes,
                                                mh_nodes)
            mh_cons_inc = self.__inc_txt_color(mh_cons_inc,
                                               self._past_max_hidden_connections,
                                               mh_cons)
            avg_fit_inc = self.__inc_txt_color(avg_fit_inc,
                                               self._past_avg_fitness, avg_fit)
            avg_fit_pc = self.__inc_txt_color(avg_fit_pc,
                                              self._past_avg_fitness, avg_fit)

        # table
        self.__table = [
            ["Best fitness", f"{b_fit:+0.{self.p}E}",
                f"{self._past_best_fitness:+0.{self.p}E}", b_fit_inc, b_fit_pc],
            ["Avg pop fitness", f"{avg_fit:+0.{self.p}E}",
                f"{self._past_avg_fitness:+0.{self.p}E}",
                avg_fit_inc, avg_fit_pc],
            ["Max hd nodes", str(mh_nodes), str(self._past_max_hidden_nodes),
                str(mh_nodes_inc), "-"],
            ["Max hd connections", str(mh_cons),
                str(self._past_max_hidden_connections), str(mh_cons_inc), "-"],
        ]

        # print
        print("done!")

        # update cache
        self._past_best_fitness = b_fit
        self._past_max_hidden_nodes = mh_nodes
        self._past_max_hidden_connections = mh_cons
        self._past_avg_fitness = avg_fit

    def on_mass_extinction_counter_updated(self, mass_ext_counter: int) -> None:
        ct = mass_ext_counter
        ct_max = self.population.config.mass_extinction_threshold

        if self.colors:
            pc = ct / ct_max
            ct = style(str(ct), fg=("green" if pc < 0.33
                                    else "yellow" if pc < 0.66
                                    else "red"), bold=(pc >= 0.66))
        print(f"Mass extinction counter: {ct} / {ct_max}\n")

    def on_mass_extinction_start(self) -> None:
        print(columnar(self.__table, CompleteStdOutLogger.__TAB_HEADER,
                       **CompleteStdOutLogger._TAB_ARGS), end="")
        msg = "Mass extinction in progress..."
        print(f"\n"
              f"{msg if not self.colors else style(msg, fg='red', bold=True)}",
              end="")

    def on_reproduction_start(self) -> None:
        # new node mutation chance
        cur_node = self.population.config.new_node_mutation_chance
        node_inc = f"{100*(cur_node - self._past_new_node_mutation):+0.{self.p}f}"

        # new connection mutation chance
        cur_con = self.population.config.new_connection_mutation_chance
        con_inc = f"{100*(cur_con - self._past_new_con_mutation):+0.{self.p}f}"

        # colors
        if self.colors:
            node_inc = self.__inc_txt_color(node_inc,
                                            self._past_new_node_mutation,
                                            cur_node)
            con_inc = self.__inc_txt_color(con_inc,
                                           self._past_new_con_mutation, cur_con)

        # table
        self.__table += [
            ["New node chance", f"{cur_node:.{self.p}%}",
                f"{self._past_new_node_mutation:.{self.p}%}", node_inc, "-"],
            ["New conn chance", f"{cur_con:0.{self.p}%}",
                f"{self._past_new_con_mutation:.{self.p}%}", con_inc, "-"],
        ]

        # print
        print(columnar(self.__table, CompleteStdOutLogger.__TAB_HEADER,
                       **CompleteStdOutLogger._TAB_ARGS))
        print("Reproduction... ", end="")

        # update cache
        self._past_new_node_mutation = cur_node
        self._past_new_con_mutation = cur_con

    def on_speciation_start(self, invalid_genomes_replaced: int) -> None:
        print("done!\n"
              f"Invalid genomes replaced: {invalid_genomes_replaced}\n"
              "Speciation... ", end="")

    def on_generation_end(self, current_generation: int,
                          total_generations: int) -> None:
        print("done!\n")
        print("#" * CompleteStdOutLogger._SEP_SIZE)


class SimpleStdOutLogger(Callback):
    """
    TODO
    """


class History(Callback):
    """ Callback that records events during the evolutionary process.

    TODO
    """

    def __init__(self):
        super().__init__()
        self._current_generation = 0

        self.best_fitness = []                # type: List[float]
        self.avg_pop_fitness = []             # type: List[float]

        self.max_hidden_nodes = []            # type: List[int]
        self.max_hidden_connections = []      # type: List[int]

        self.species = []                     # todo
        self.extinction_counter = []          # type: List[int]
        self.extinction_events = []           # type: List[int]
        self.invalid_genomes_replaced = []    # type: List[int]

        self.new_node_chance = []             # type: List[float]
        self.new_connection_chance = []       # type: List[float]
        self.weight_mutation_chance = []      # type: List[float]
        self.enable_connection_chance = []    # type: List[float]
        self.weight_perturbation_chance = []  # type: List[float]
        self.weight_reset_chance = []         # type: List[float]

    def on_generation_start(self,
                            current_generation: int,
                            total_generations: int) -> None:
        self._current_generation = current_generation

    def on_fitness_calculated(self,
                              best_genome: neat.genome.Genome,
                              max_hidden_nodes: int,
                              max_hidden_connections: int) -> None:
        self.best_fitness.append(best_genome.fitness)
        self.avg_pop_fitness.append(float(
            np.mean([g.fitness for g in self.population.genomes])
        ))
        self.max_hidden_nodes.append(max_hidden_nodes)
        self.max_hidden_connections.append(max_hidden_connections)

    def on_mass_extinction_counter_updated(self,
                                           mass_ext_counter: int) -> None:
        self.extinction_counter.append(mass_ext_counter)

    def on_mass_extinction_start(self) -> None:
        self.extinction_events.append(self._current_generation)

    def on_reproduction_start(self) -> None:
        cfg = self.population.config
        self.new_node_chance.append(cfg.new_node_mutation_chance)
        self.new_connection_chance.append(cfg.new_connection_mutation_chance)
        self.weight_mutation_chance.append(cfg.weight_mutation_chance)
        self.enable_connection_chance.append(
            cfg.enable_connection_mutation_chance)
        self.weight_perturbation_chance.append(cfg.weight_perturbation_pc)
        self.weight_reset_chance.append(cfg.weight_reset_chance)

    def on_speciation_start(self, invalid_genomes_replaced: int) -> None:
        self.invalid_genomes_replaced.append(invalid_genomes_replaced)

    def on_generation_end(self,
                          current_generation: int,
                          total_generations: int) -> None:
        pass

    def visualize(self):
        """
        TODO
        """


class PopulationCheckpoint(Callback):
    """
    TODO
    """


class BestGenomeCheckpoint(Callback):
    """
    TODO
    """
