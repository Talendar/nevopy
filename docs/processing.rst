==========
Processing
==========

------------
Introduction
------------

In `NEvoPy`, most of the heavy processing involved in evolving a population of
neural networks is managed by a `processing scheduler`. Processing schedulers
allow the implementation of the computation methods (like the use of serial or
parallel processing) to be separated from the implementation of the
neuroevolutionary algorithms. Examples of processing schedulers in `NEvoPy`
include the :class:`.PoolProcessingScheduler`, that uses Python's
:mod:`multiprocessing` module to implement parallel processing, and the
:class:`.RayProcessingScheduler`, that uses the :mod:`ray` framework to
implement distributed computing (it even allows you to use clusters!).

In this quick guide you'll learn what a `NEvoPy` processing scheduler is, what
it can do, and how you can build your own. A demo of a simple application of
processing schedulers is provided to get you started.

For a list with all the pre-implemented processing schedulers, take a look at
:mod:`nevopy.processing`.


---------------------------------------
`NEvoPy` processing schedulers overview
---------------------------------------

todo


--------------------------------------
Writing your own processing schedulers
--------------------------------------

todo (multi-genome fitness function)