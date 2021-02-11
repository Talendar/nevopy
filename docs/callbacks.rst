=========
Callbacks
=========

------------
Introduction
------------

A callback is a powerful tool to customize the behaviour of a population of
genomes during the neuroevolutionary process. Examples include
:class:`.FitnessEarlyStopping` to stop the evolution when a certain fitness has
been achieved by the population, or :class:`.BestGenomeCheckpoint` to
periodically save the best genome of a population during evolution. For a list
with all the pre-implemented callbacks, take a look at :mod:`.callbacks`.

In this quick guide you'll learn what a `NEvoPy` callback is, what it can do,
and how you can build your own.

---------------------------
`NEvoPy` callbacks overview
---------------------------

In `NEvoPy`, all callbacks subclass the :class:`.Callback` class and override a
set of methods called at various stages of the evolutionary process. Callbacks
are useful to get a view on internal states and statistics of a population and
its genomes, as well as for modifying the behavior of the evolutionary algorithm
being used.

You can pass a list of callbacks (as the keyword argument ``callbacks``) to
the :meth:`evolve() <.BasePopulation.evolve()>` method of your population.


-------------------------------
An overview of callback methods
-------------------------------

A callback implements one or more of the following methods (check each method's
documentation for a list of accepted parameters):

    * :meth:`on_generation_start <.Callback.on_generation_start>`: called at the
      beginning of each new generation.
    * :meth:`on_fitness_calculated <.Callback.on_fitness_calculated>`: called
      right after the fitness values of the population's genomes are calculated.
    * :meth:`on_mass_extinction_counter_updated
      <.Callback.on_mass_extinction_counter_updated>`: called right after the
      mass extinction counter is updated. The mass extinction counter counts how
      many generations have passed since the fitness of the population's best
      genomes improved.
    * :meth:`on_mass_extinction_start <.Callback.on_mass_extinction_start>`:
      called at the beginning of a mass extinction event. A mass extinction
      event occurs when the population's fitness haven't improved for a
      predefined number of generations. It results in the replacement of all the
      population's genomes (except for the fittest genome) for new randomly
      generated genomes.
    * :meth:`on_reproduction_start <.Callback.on_reproduction_start>`: called
      at the beginning of the reproductive process.
    * :meth:`on_speciation_start <.Callback.on_speciation_start>`: called at
      the beginning of the speciation process. If the neuroevolutionary
      algorithm doesn't use speciation, this method isn't called at all.
    * :meth:`on_generation_end <.Callback.on_generation_end>`: called at the
      end of each generation.
    * :meth:`on_evolution_end <.Callback.on_evolution_end>`: called when the
      evolutionary process ends.


--------------------------
Writing your own callbacks
--------------------------

To build your own callback, simply create a new class that has
:class:`.Callback` as its parent class:

    .. code-block:: python

        class MyCallback(Callback):
            def on_generation_start(self,
                                    current_generation,
                                    max_generations):
                print("This is printed at the start of every generation!")
                print(f"Starting generation {current_generation} of "
                      f"{max_generations}.")

Then, just create a new instance of your callback and pass it to the
:meth:`evolve() <.BasePopulation.evolve()>` of your population:

    .. code-block:: python

        population.evolve(generations=100,
                          fitness_function=my_func,
                          callbacks=[MyCallback()])
