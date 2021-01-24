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
                                    max_generations):
                print("This is printed at the start of every generation!")
                print(f"Starting generation {current_generation} of "
                      f"{max_generations}.")

        pop.evolve(generations=100,
                   fitness_function=my_func,
                   callbacks=[MyCallback()])
"""

from typing import Optional, List, Dict, Union, Tuple, Callable
from nevopy import neat
from nevopy import utils

import numpy as np
import matplotlib.pyplot as plt

from columnar import columnar
from click import style
from timeit import default_timer as timer


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
                            max_generations: int) -> None:
        """ Called at the beginning of each new generation.

        Subclasses should override this method for any actions to run.

        Args:
            current_generation (int): Number of the current generation.
            max_generations (int): Maximum number of generations.
        """

    def on_fitness_calculated(self,
                              best_genome: neat.genomes.NeatGenome,
                              max_hidden_nodes: int,
                              max_hidden_connections: int) -> None:
        """ Called right after the fitness of the population's genomes and the
        maximum number of hidden nodes and hidden connections in the population
        are calculated.

        Subclasses should override this method for any actions to run.

        Args:
            best_genome (NeatGenome): The most fit genome of the generation.
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
                          max_generations: int) -> None:
        """ Called at the end of each generation.

        Subclasses should override this method for any actions to run.

        Args:
            current_generation (int): Number of the current generation.
            max_generations (int): Maximum number of generations.
        """

    def on_evolution_end(self, total_generations: int) -> None:
        """ Called when the evolutionary process ends.

        Args:
            total_generations (int): Total number of generations processed
                during the evolutionary process. Might not be the maximum number
                of generations specified by the user, if some sort of early
                stopping occurs.
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

    def __init__(self,
                 precision: int = 2,
                 colors: bool = True,
                 output_cleaner: Optional[Callable] = utils.clear_output):
        super().__init__()
        self.p = precision
        self.clear_output = output_cleaner
        self.colors = colors
        self._past_num_species = 0
        self._past_best_fitness = 0.0
        self._past_avg_fitness = 0.0
        self._past_max_hidden_nodes = 0
        self._past_max_hidden_connections = 0
        self._past_new_node_mutation = 0.0
        self._past_new_con_mutation = 0.0
        self.__msg_cache = ""
        self.__table = None  # type: Optional[List[List[str]]]
        self._timer = 0.0

    @staticmethod
    def __inc_txt_color(txt, past, current):
        return style(txt,
                     fg=(CompleteStdOutLogger._POS_COLOR if current > past
                         else CompleteStdOutLogger._NEG_COLOR if current < past
                         else CompleteStdOutLogger._NEUT_COLOR))

    def on_generation_start(self,
                            current_generation: int,
                            max_generations: int) -> None:
        self._timer = timer()
        g_cur, g_tot = current_generation, max_generations
        sp_cur = len(self.population.species)
        sp_inc = f"{sp_cur - self._past_num_species:+0d}"

        if self.colors:
            sp_inc = self.__inc_txt_color(sp_inc,
                                          self._past_num_species, sp_cur)

        print("\n"
              f"[{(g_cur + 1) / g_tot:.2%}] "
              f"Generation {g_cur + 1} of {g_tot}.\n"
              f". Number of species: {sp_cur} ({sp_inc})")
        print(f". Calculating fitness "
              f"(last: {self._past_best_fitness:.{self.p}E})... ", end="")

        self.__msg_cache += f". Number of species: {sp_cur} ({sp_inc})\n"
        self._past_num_species = sp_cur

    def on_fitness_calculated(self,
                              best_genome: neat.genomes.NeatGenome,
                              max_hidden_nodes: int,
                              max_hidden_connections: int) -> None:
        # best fitness
        b_fit = best_genome.fitness
        b_fit_inc_float = b_fit - self._past_best_fitness
        b_fit_pc_float = (float("inf") if self._past_best_fitness == 0
                          else b_fit_inc_float/self._past_best_fitness)
        b_fit_pc_float = abs(b_fit_pc_float) * (1, -1)[b_fit_inc_float < 0]

        b_fit_inc = f"{b_fit_inc_float:+0.{self.p}E}"
        b_fit_pc = f"{b_fit_pc_float:+0.{self.p}%}"

        # max hidden nodes
        mh_nodes = max_hidden_nodes
        mh_nodes_inc = f"{mh_nodes - self._past_max_hidden_nodes:+0d}"

        # max hidden connections
        mh_cons = max_hidden_connections
        mh_cons_inc = f"{mh_cons - self._past_max_hidden_connections:+0d}"

        # avg pop fitness
        avg_fit = np.mean([g.fitness for g in self.population.genomes])
        avg_fit_inc_float = avg_fit - self._past_avg_fitness
        avg_fit_pc_float = (float("inf") if self._past_avg_fitness == 0
                            else avg_fit_inc_float / self._past_avg_fitness)
        avg_fit_pc_float = abs(avg_fit_pc_float) * (1, -1)[avg_fit_inc_float < 0]

        avg_fit_inc = f"{avg_fit_inc_float:+0.{self.p}E}"
        avg_fit_pc = f"{avg_fit_pc_float:+0.{self.p}%}"

        # colors
        if self.colors:
            b_fit_inc = self.__inc_txt_color(b_fit_inc,
                                             self._past_best_fitness, b_fit)
            b_fit_pc = self.__inc_txt_color(b_fit_pc,
                                            self._past_best_fitness, b_fit)
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
        ct = mass_ext_counter  # type: Union[int, str]
        ct_max = self.population.config.mass_extinction_threshold

        if self.colors:
            pc = ct / ct_max
            ct = style(str(ct), fg=("green" if pc < 0.33
                                    else "yellow" if pc < 0.66
                                    else "red"), bold=(pc >= 0.66))
        print(f". Mass extinction counter: {ct} / {ct_max}")
        self.__msg_cache += f". Mass extinction counter: {ct} / {ct_max}\n"

    def on_mass_extinction_start(self) -> None:
        msg = ". Mass extinction in progress... "
        print(f"{msg if not self.colors else style(msg, fg='red', bold=True)}",
              end="")

        msg = '. Mass extinction occurred!'
        c = f"{msg if not self.colors else style(msg, fg='red', bold=True)}\n"
        self.__msg_cache += ". " + c

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
        print(". Reproduction... ", end="")

        # update cache
        self._past_new_node_mutation = cur_node
        self._past_new_con_mutation = cur_con

    def on_speciation_start(self, invalid_genomes_replaced: int) -> None:
        print("done!\n"
              f". Invalid genomes replaced: {invalid_genomes_replaced}\n"
              ". Speciation... ", end="")
        self.__msg_cache += (". Invalid genomes replaced: "
                             f"{invalid_genomes_replaced}")

    def on_generation_end(self,
                          current_generation: int,
                          max_generations: int) -> None:
        print("done!\n")
        if self.clear_output is not None:
            self.clear_output()

        print(f">> GENERATION {current_generation + 1} SUMMARY:")
        if self.clear_output is not None:
            print("\n" + self.__msg_cache)

        print(f". Processing time: {timer() - self._timer : .4f}s\n")
        print(columnar(self.__table, CompleteStdOutLogger.__TAB_HEADER,
                       **CompleteStdOutLogger._TAB_ARGS))
        print("#" * CompleteStdOutLogger._SEP_SIZE)
        self.__msg_cache = ""

    def on_evolution_end(self, total_generations: int) -> None:
        print(f"Evolution ended after {total_generations + 1} generations.")


class SimpleStdOutLogger(Callback):
    """
    TODO
    """


class History(Callback):
    """ Callback that records events during the evolutionary process.

    Attributes:
        best_fitness (List[float]): List with the fitness of the best genome
            of each generation.
        avg_pop_fitness (List[float]): List with the average fitness of the
            population in each generation.
        max_hidden_nodes (List[int]): The maximum number of hidden nodes present
            in a genome in the population in each generation.
        max_hidden_connections (List[int]): The maximum number of hidden
            connections (connections linked to at least one hidden node) in a
            genome in the population in each generation.
        species_info (Dict[int, Dict]): Contains information about all the
            species that emerged during the evolution. It's a dictionary that
            maps a species ID (an int) to another dictionary containing the
            following keys: "born" (generation in which the species was born);
            "extinct" (generation in which the species was extinct or `None` if
            it wasn't extinct); "size" (list with the number of genomes in the
            species in each generation); "best_fitness" (list with the fitness
            of the best genome in the species in each generation); "avg_fitness"
            (list with the average fitness of the species in each generation).
        num_species (List[int]): List with the number of species alive in each
            generation.
        extinction_counter (List[int]): List with the value of the extinction
            counter in each generation.
        extinction_events (List[int]): List with the generations in which an
            extinction event occurred.
        invalid_genomes_replaced (List[int]): List with the number of invalid
            genomes replaced in each generation.
        new_node_chance (List[float]): List with the chance of a new hidden node
            mutation in each generation.
        new_connection_chance (List[float]): List with the chance of a new
            hidden connection mutation in each generation.
        weight_mutation_chance (List[float]): List with the chance of a weight
            mutation in each generation.
        enable_connection_chance (List[float]): List with the chance of a enable
            connection mutation in each generation.
        weight_perturbation_chance (List[float]): List with the chance of a
            weight perturbation mutation in each generation.
        weight_reset_chance (List[float]): List with the chance of a weight
            reset mutation in each generation.
    """

    PLOT_ATTRS = ("best_fitness", "avg_pop_fitness", "num_species",
                  "max_hidden_nodes", "max_hidden_connections",
                  "extinction_counter", "new_node_chance",
                  "new_connection_chance", "invalid_genomes_replaced",
                  "weight_mutation_chance", "enable_connection_chance",
                  "weight_perturbation_chance", "weight_reset_chance")

    def __init__(self):
        super().__init__()
        self._current_generation = 0

        self.best_fitness = []                # type: List[float]
        self.avg_pop_fitness = []             # type: List[float]

        self.max_hidden_nodes = []            # type: List[int]
        self.max_hidden_connections = []      # type: List[int]

        self.species_info = {}                # type: Dict[int, Dict]
        self.num_species = []                 # type: List[int]

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
                            max_generations: int) -> None:
        self._current_generation = current_generation

        # adding pioneer species
        if len(self.species_info) == 0:
            for sid, sp in self.population.species.items():
                self._new_species_info(sid, current_generation, len(sp.members))

    def on_fitness_calculated(self,
                              best_genome: neat.genomes.NeatGenome,
                              max_hidden_nodes: int,
                              max_hidden_connections: int) -> None:
        # recording best and avg fitnesses
        self.best_fitness.append(best_genome.fitness)
        self.avg_pop_fitness.append(float(
            np.mean([g.fitness for g in self.population.genomes])
        ))

        # recording max num of hidden nodes and connections
        self.max_hidden_nodes.append(max_hidden_nodes)
        self.max_hidden_connections.append(max_hidden_connections)

        # updating species fitness
        for sid, sp in self.population.species.items():
            self.species_info[sid]["best_fitness"].append(sp.fittest().fitness)
            self.species_info[sid]["avg_fitness"].append(sp.avg_fitness())

    def on_mass_extinction_counter_updated(self,
                                           mass_ext_counter: int) -> None:
        self.extinction_counter.append(mass_ext_counter)

    def on_mass_extinction_start(self) -> None:
        self.extinction_events.append(self._current_generation)

    def on_reproduction_start(self) -> None:
        # recording new mutation chances
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
                          max_generations: int) -> None:
        # num species
        self.num_species.append(len(self.population.species))

        # checking extinct species and updating species size
        for sid, info in self.species_info.items():
            if sid not in self.population.species:
                if info["extinct"] is None:
                    info["extinct"] = current_generation
                    info["size"].append(0)
            else:
                info["size"].append(len(self.population.species[sid].members))

        # adding new species
        for sid, sp in self.population.species.items():
            if sid not in self.species_info:
                self._new_species_info(sid, current_generation, len(sp.members))

    def _new_species_info(self, sid, gen, size):
        assert sid not in self.species_info
        self.species_info[sid] = {"born": gen, "extinct": None, "size": [size],
                                  "best_fitness": [], "avg_fitness": []}

    def visualize(self,
                  attrs: Union[Tuple[str, ...], str] = ("best_fitness",
                                                        "avg_pop_fitness"),
                  figsize=(10, 6),
                  log_scale: bool = True) -> None:
        """ Simple utility method for plotting the recorded information.

        This method is a simple wrapper around `matplotlib`. It isn't suited for
        advanced plotting.

        Attributes:
            attrs (Union[Tuple[str, ...], str]): Tuple with the names of the
                attributes to be plotted. If "all", all plottable attributes are
                plotted.
            log_scale (bool): Whether or not to use a logarithmic scale on the
                y-axis.
        """
        generations = range(len(self.best_fitness))
        if type(attrs) is str and attrs == "all":
            attrs = History.PLOT_ATTRS

        plt.figure(figsize=figsize)
        plt.yscale("log" if log_scale else "linear")
        for key in attrs:
            data = self.__dict__[key]
            plt.plot(generations, data, label=key)

        plt.legend()
        plt.show()

    def visualize_species(self) -> None:
        """ TODO

        """
        raise NotImplementedError


class FitnessEarlyStopping(Callback):
    """ Stops the evolution if a given fitness value is achieved.

    This callback is used to halt the evolutionary process when a certain
    fitness value is achieved by the population's best genome for a given number
    of consecutive generations.

    Args:
        fitness_threshold (float): Fitness to be achieved for the evolution to
            stop.
        min_consecutive_generations (int): Number of consecutive generations
            with a fitness equal or higher than ``fitness_threshold`` for the
            early stopping occur.

    Attributes:
        fitness_threshold (float): Fitness to be achieved for the evolution to
            stop.
        min_consecutive_generations (int): Number of consecutive generations
            with a fitness equal or higher than ``fitness_threshold`` for the
            early stopping to occur.
        stopped_generation (Optional[int]): Generation in which the early
            stopping occurred. `None` if the early stopping never occurred.
    """

    def __init__(self,
                 fitness_threshold: float,
                 min_consecutive_generations: int) -> None:
        super().__init__()
        self.fitness_threshold = fitness_threshold
        self.min_consecutive_generations = min_consecutive_generations
        self.stopped_generation = None  # type: Optional[int]
        self._consec_gens = 0

    def on_fitness_calculated(self,
                              best_genome: neat.genomes.NeatGenome,
                              max_hidden_nodes: int,
                              max_hidden_connections: int) -> None:
        if best_genome.fitness >= self.fitness_threshold:
            self._consec_gens += 1
        else:
            self._consec_gens = 0

    def on_generation_end(self,
                          current_generation: int,
                          max_generations: int) -> None:
        if self._consec_gens >= self.min_consecutive_generations:
            self.population.stop_evolving = True
            self.stopped_generation = current_generation


class PopulationCheckpoint(Callback):
    """
    TODO
    """


class BestGenomeCheckpoint(Callback):
    """
    TODO
    """
