===============
NEvoPy Overview
===============

---------------------
Neuroevolution basics
---------------------

Neuroevolution refers to the artificial evolution of neural networks using
evolutionary algorithms. It's heavily inspired by the biological concept of
`Evolution <https://en.wikipedia.org/wiki/Evolution>`_ and makes use of a
population-based metaheuristic and mechanisms such as selection, reproduction,
recombination and mutation to generate solutions.

A neural network is encoded, either directly or indirectly, by a `genome` (also
called `genotype` or `individual`). The neural network encoded by a genome is
its `phenotype`. We call a set of competing genomes a `population`. A genome's
`fitness` is a measure of how well the genome performs in a given task. The goal
of a neuroevolutionary algorithm is to evolve a population of genomes in order
to produce genomes with a high fitness value.

The evolutionary process is divided into generations. In each generation, the
population's genomes have their `fitness` calculated. Genomes with a higher
fitness value have a greater chance of leaving offspring for the next
generation. By favoring the reproduction of fitter genomes, the algorithm
gradually increases the total fitness of the population.

If you are a beginner to neuroevolution and want to know more about this
awesome area of research, here's a couple of papers and articles to get you
started:

    * `Evolving artificial neural networks <https://ieeexplore.ieee.org/document/784219>`_
      (great review paper);
    * `Evolving Neural Networks through Augmenting Topologies <http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf>`_
      (the original paper of the NEAT algorithm);
    * `Neuroevolution: A different kind of deep learning <https://www.oreilly.com/radar/neuroevolution-a-different-kind-of-deep-learning/>`_
      (great introductory article about NE, by the creator of NEAT);
    * `Neuroevolution: A Primer On Evolving Artificial Neural Networks <https://www.inovex.de/blog/neuroevolution/>`_
      (great introductory article about NE);
    * `Welcoming the Era of Deep Neuroevolution <https://eng.uber.com/deep-neuroevolution/>`_
      (article about recent research by Uber AI Labs).


-----------------------------------
Populations and genomes in `NEvoPy`
-----------------------------------

In `NEvoPy`, a genome is an instance of a subclass that implements
:class:`.BaseGenome`. Although each neuroevolutionary algorithm defines its own
type of genome by implementing the :class:`.BaseGenome` class, all genomes are
governed by the same general API. Note that in `NEvoPy's` API there isn't any
distinction between a genome and the neural network it encodes. A genome, just
like a neural network, must be capable of processing inputs based on its nodes
and connections in order to produce an output. It also must be able to mutate
and to generate offspring.

A population of genomes, on the other hand, is represented by the class
:class:`.BasePopulation`. It defines a general API that all neuroevolutionary
algorithms implemented by `NEvoPy` follow. Each algorithm makes its own
implementation of that class - it's where the core of the evolutionary algorithm
lives. The main method of the API is :meth:`.BasePopulation.evolve()`, which
triggers the evolutionary process in a population.

Most neuroevolutionary algorithms use a genetic algorithm to evolve the neural
networks. What usually changes between different algorithms is how the genomes
behave (how they reproduce, mutate and encode a neural network, for example).
With that in mind, `NEvoPy` implements a general-purpose genetic
algorithm (see :class:`.GeneticPopulation`) that can be used as a base for most
neuroevolutionary algorithms. This algorithm doesn't make strong assumptions
about the genomes its evolving (it "doesn't care" if the genome encodes a
network directly or indirectly, for example), so it can be used in a wide
variety of scenarios. It also supports speciation.

`NEvoPy` currently implements the following neuroevolutionary algorithms:

    * :doc:`Neuroevolution of Augmenting Topologies (NEAT) <nevopy.neat>`;
    * :doc:`Fixed-topology deep-neuroevolution <nevopy.fixed_topology>`.

However, if you need more, implementing your own neuroevolutionary algorithm
with `NEvoPy` is easy. Simply create a class that implements
:class:`.BaseGenome` (thus defining how you want your genomes to behave) and let
:class:`.GeneticPopulation` do the rest.


--------------------------------------
Evolving neural networks with `NEvoPy`
--------------------------------------

To evolve some neural networks with `NEvoPy`, the first thing you have to do is
create a new population of genomes (represented by a class that implements
:class:`.BasePopulation`). As an example, let's create a
:class:`.NeatPopulation` (implements the NEAT algorithm):

    .. code-block:: python

            import nevopy as ne
            population = ne.neat.NeatPopulation(size=100,
                                                num_inputs=10,
                                                num_outputs=3)

The code above creates an instance of :class:`.NeatPopulation`, used to evolve
instances of :class:`.NeatGenome` with the NEAT algorithm. The genomes are built
to receive an array-like input of length 10 and to output the results as an
array-like object of length 3. In `NEvoPy`, the inputs and outputs are, in most
cases, instances of :class:`numpy.ndarray` or :class:`tensorflow.tensor`.

Now, we need to specify some routine for evaluating the population's genomes,
i.e., for measuring the performance of each of the population's genomes on the
task at hand in each `generation`. We call the measure of a genome's performance
its `fitness` and the routine used to calculate this value a `fitness function`.
Generally, a fitness function should look like this:

    .. code-block:: python

        def fitness_function(genome):
            # (the genome's fitness is calculated here)
            # ...
            return fitness


Having created a population and defined a fitness function, we're ready to start
the evolutionary process. We do that by calling the
:meth:`evolve() <.BasePopulation.evolve>` method:

    .. code-block:: python

        history = population.evolve(generations=100,
                                    fitness_function=fitness_function)

The code above runs the NEAT algorithm for 100 generations. The
:meth:`evolve() <.BasePopulation.evolve>` method returns a :class:`.History`
object, which contains useful statistics related to the evolutionary process. We
can, for example, visualize the progression of the population's fitness by
executing the following:

    .. code-block:: python

        history.visualize()

Here is an example of a plot generated by this method:

.. figure:: /imgs/fitness_history_sample.png

The code bellow gets the fittest genome of the population, visualizes its
topology and saves the genome:

    .. code-block:: python

        best_genome = population.fittest()
        best_genome.visualize()
        best_genome.save("./best_genome.pkl")

For more information on how `NEvoPy` works, please take a look at our
:doc:`docs <index>`. For more practical examples, go to :doc:`here <examples>`.
