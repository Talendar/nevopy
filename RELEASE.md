## Release 0.1.1

* fix gym utils idx/argmax problem
* update fix top genome visualize to work with colab
* add colab xor example
* add wrapper for tf.keras.layers.MaxPool2D


## Release 0.1.0

### Breaking Changes

* `ne.fixed_topology.FixedTopologyPopulation` has been deprecated. The new class
  `ne.genetic_algorithm.GeneticPopulation` should be
  used in its place.
* `ne.fixed_topology.FixedTopologyConfig` has been  deprecated. The new class
  `ne.genetic_algorithm.GeneticAlgorithmConfig` should used in its place.  

### Bug Fixes and Other Changes

* Added new type of population: `ne.genetic_algorithm.GeneticPopulation`. It
  implements a generalizable genetic algorithm that can be used as a base for a
  wide range of neuroevolutionary algorithms. This resolves
  [#1](https://github.com/Talendar/nevopy/issues/1).
* Added `deprecation.py` to `utils`. It implements the `@deprecated` decorator,
  that can be used to mark a function, method or class as being deprecated.
* Fixed a bug in `GymEnvFitness` that was causing an incorrect interpretation of
  the output values of fixed-topology genomes using TensorFlow layers.
* Made some fixes and additions to the project's docstrings.
* Added a new example in which *NEvoPy* is used to create an AI to play the
  *Flappy Bird* game.
* The other examples were reformatted and comments explaining the code were
  added to them.


## Release 0.0.2

Initial public release of the *NEvoPy* framework.