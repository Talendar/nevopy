## Release 0.2.3

### Bug Fixes and Other Changes

* Fixed `nevopy.callbacks.BestGenomeCheckpoint` not working properly with
  negative fitness values.
* Fixed `nevopy.neat.visualization.columns_graph_layout` wrongfully positioning
  hidden nodes when the network had only one column of hidden nodes.
  
### Breaking Changes

* Removed the `file_prefix` parameter from the
  `nevopy.callbacks.BestGenomeCheckpoint.__init__` method.
* Changed the default values of the parameters `out_path` and
  `min_improvement_pc` of the `nevopy.callbacks.BestGenomeCheckpoint.__init__`
  method.


## Release 0.2.2

### Bug Fixes and Other Changes

* Added the `gym` package to the project's dependencies (version 0.17.3).
* Fixed missing docs on RTD due to dependencies issues.

NOTE: the version 0.2.1 was released as a quick-fix for PyPI only.


## Release 0.2.0

### Major Features and Improvements

* Added a new function (`nevopy.neat.visualization.visualize_activations`) to
  visualize the neural topology of NEAT genomes (`nevopy.neat.NeatGenome`) while
  they interact with an environment. It's highly customizable and allows, among
  other things, activated nodes and edges to be drawn with different colors.
  Check the lunar lander example for a demonstration.
* Added new utilities to be used with `gym` environments, including callbacks
  and custom renderers. They're all contained within the new subpackage
  `nevopy.utils.gym_utils`.
* Added the bipedal walker example.
* Improved the lunar lander example. Now it shows the neural topology of the
  evolved genome side by side with the rendering of the environment.

### Breaking Changes

* Removed `nevopy.fixed_topology.FixedTopologyPopulation` (deprecated since
  v0.1.0). The `nevopy.genetic_algorithm.GeneticPopulation` class should be 
  used instead.
* Removed `nevopy.fixed_topology.FixedTopologyConfig` (deprecated since v0.1.0).
  The `nevopy.genetic_algorithm.GeneticAlgorithmConfig` class should be used
  instead.  
* Removed the Mario example.
* Replaced `nevopy.utils.GymEnvFitness` with `nevopy.utils.GymFitnessFunction`,
  which has more features and supports more advanced callbacks.


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
