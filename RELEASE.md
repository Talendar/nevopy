## Release 0.1.1

### Bug Fixes and Other Changes

* Fixed a bug in `ne.utils.GymEnvFitness` that led to an incorrect
  interpretation of the agent's chosen action when dealing with TensorFlow
  tensors.
* Updated `ne.fixed_topology.FixedTopologyGenome.visualize`. It now returns the
  generated `PIL.Image.Image` object. It's possible now to directly use the
  method to visualize the genome's topology on a Jupyter Notebook.
* Added a new XOR example (Jupyter Notebook).
* Added a wrapper to `tf.keras.layers.MaxPool2D` in `ne.fixed_topology.layers`.
* Fixed a bug that occurred when a string was passed to the ``layer_type``
  parameter of the constructor of the `ne.fixed_topology.TensorFlowLayer` class.


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