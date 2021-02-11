==========
Processing
==========

------------
Introduction
------------

In `NEvoPy`, most of the heavy processing involved in evolving a population of
neural networks is managed by a `processing scheduler`. Processing schedulers
allow the implementation of computation methods (like the use of serial or
parallel processing) to be separated from the implementation of the
neuroevolutionary algorithms. Examples of processing schedulers in `NEvoPy`
include the :class:`.PoolProcessingScheduler`, that uses Python's
:mod:`multiprocessing` module to implement parallel processing, and the
:class:`.RayProcessingScheduler`, that uses the :mod:`ray` framework to
implement distributed computing (it even allows you to use clusters!).

In this quick guide you'll learn what a `NEvoPy` processing scheduler is, what
it can do, and how you can build your own. For a list with all the
pre-implemented processing schedulers, take a look at :mod:`nevopy.processing`.


---------------------------------------
`NEvoPy` processing schedulers overview
---------------------------------------

In `NEvoPy`, all processing schedulers subclass the
:class:`.ProcessingScheduler` class and override its
:meth:`run() <.ProcessingScheduler.run>` method, which is responsible for
processing a batch of items (:attr:`TProcItem <.base_scheduler.TProcItem>`)
and returning the corresponding results
(:attr:`TProcResult <.base_scheduler.TProcResult>`). The items and the results
can be anything, but they usually are genomes and their fitnesses, respectively.

Processing schedulers might also be used to handle the computations associated
with the reproductive process of a population.


--------------------------------------
Writing your own processing schedulers
--------------------------------------

To build your own processing scheduler, simply create a new class that has
:class:`.ProcessingScheduler` as its parent class and implement the
:meth:`run() <.ProcessingScheduler.run>` method:

    .. code-block:: python

        class MyProcessingScheduler(ProcessingScheduler):

            def run(items, func):
                # ...
                return results

Then, just create a new instance of your new processing scheduler and pass it to
the constructor of your population!