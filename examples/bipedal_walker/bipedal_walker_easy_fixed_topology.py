""" In this example, we're going to use NEvoPy's implementation of a classic
fixed-topology neuroevolution algorithm to learn how to walk on two legs!

We're going to use the "BipedalWalker-v3" gym environment to run the
simulations.

Authors:
    Gabriel Nogueira (Talendar)
"""

import os

import gym
import nevopy as ne


# Setting this environment variable to "true" will avoid a single process from
# allocating all the GPU memory by itself. Since NEvoPy heavily relies on
# multi-processing, this step is important if you're going to use a GPU.
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Defining a base genome for our population. It will serve as a model for the
# other genomes of the population, which will have the same topology as it.
BASE_GENOME = ne.fixed_topology.FixedTopologyGenome(

    # List with the layers of the base genome. Other genomes in the population
    # will have similar layers (with the same topology).
    layers=[ne.fixed_topology.layers.TFDenseLayer(32, activation="relu"),
            ne.fixed_topology.layers.TFDenseLayer(4, activation="tanh")],

    # Shape of the input samples expected by the genome.
    input_shape=[1, 24],
)


def make_env():
    """ Makes a new gym "BipedalWalker-v3" environment. """
    return gym.make("BipedalWalker-v3")


if __name__ == "__main__":
    # NEvoPy provides a general fitness function that works with most gym
    # environments, so we don't have to worry about implementing it.
    fitness_function = ne.utils.GymFitnessFunction(
        make_env=make_env,

        # Number of "tries" the agent will have in each playing session:
        default_num_episodes=1,

        # Since we're using TensorFlow layers, which require inputs in batches,
        # we must change the shape of the input sample before feeding it to our
        # genomes. The callback below handles that for us:
        callbacks=[ne.utils.BatchObsGymCallback()]
    )

    # Creating a new random population of fixed-topology genomes:
    # (we're going to use the default settings)
    population = ne.genetic_algorithm.GeneticPopulation(size=50,
                                                        base_genome=BASE_GENOME)

    # We can tell NEvoPy to halt the evolutionary process when a certain fitness
    # value has been achieved by the population. For that, we use a specific
    # type of callback.
    early_stopping_cb = ne.callbacks.FitnessEarlyStopping(
        fitness_threshold=300,
        min_consecutive_generations=3,
    )

    # Evolving the population (this might take some minutes):
    history = population.evolve(generations=50,
                                fitness_function=fitness_function,
                                callbacks=[early_stopping_cb])

    # Retrieving the best genome in the population:
    best_genome = population.fittest()

    # Visualizing the topology of our best genome:
    best_genome.visualize()

    # Visualizing the progression of the population's fitness:
    history.visualize(log_scale=False)

    # Now, let's compare the performance of our best genome with the performance
    # of a random agent.
    print("\nEvaluation (25 episodes):")
    print(f". Random genome score: {fitness_function(None, num_eps=25)}")
    print(f". Evolved genome: {fitness_function(best_genome, num_eps=25)}")

    # Visualizing our best genome and a random agent playing:
    while True:
        key = int(input("\nEvolution complete!\n"
                        "   [0] Visualize random genome\n"
                        "   [1] Visualize evolved genome\n"
                        "   [2] Exit\n"
                        "Choose an option: "))
        if key == 0:
            print(f"\nRandom genome: "
                  f"{fitness_function(None, num_eps=1, visualize=True)}")
        elif key == 1:
            print(f"\nEvolved genome: "
                  f"{fitness_function(best_genome, num_eps=1, visualize=True)}")
        elif key == 2:
            break
        else:
            print("\nInvalid key!")
