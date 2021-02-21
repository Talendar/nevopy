""" In this example, we're going to use NEvoPy's implementation of the NEAT
algorithm to learn how to walk on two legs!

We're going to use the "BipedalWalker-v3" gym environment to run the
simulations.

Authors:
    Gabriel Nogueira (Talendar)
"""

import gym
import nevopy as ne
import numpy as np


# Some of the settings we can change to customize the behaviour of the NEAT
# algorithm (to see all the available settings, check out the documentation of
# the class nevopy.neat.config.NeatConfig):
CONFIG = ne.neat.NeatConfig(
    weak_genomes_removal_pc=0.7,
    weight_mutation_chance=(0.7, 0.9),
    new_node_mutation_chance=(0.1, 0.5),
    new_connection_mutation_chance=(0.08, 0.5),
    enable_connection_mutation_chance=(0.08, 0.5),
    disable_inherited_connection_chance=0.75,
    mating_chance=0.75,
    interspecies_mating_chance=0.05,
    rank_prob_dist_coefficient=1.75,
    # weight mutation specifics
    weight_perturbation_pc=(0.05, 0.4),
    weight_reset_chance=(0.05, 0.4),
    new_weight_interval=(-2, 2),
    # mass extinction
    mass_extinction_threshold=10,
    maex_improvement_threshold_pc=0.03,
    # infanticide
    infanticide_output_nodes=False,
    infanticide_input_nodes=False,
    # speciation
    species_distance_threshold=0.75,
    # nodes activation
    out_nodes_activation=np.tanh,
    hidden_nodes_activation=ne.activations.steepened_sigmoid,
)


def make_env():
    """ Makes a new gym "BipedalWalker-v3" environment. """
    return gym.make("BipedalWalker-v3")


class PreProcessObsGymCallback(ne.utils.GymCallback):
    """ The last 10 features in a observation yielded by the "BipedalWalker-v3"
    environment are LIDAR rangefinder measurements, which aren't much useful to
    an agent learning how to solve the easy version of the env, as pointed in
    the environment's docs. This callback will remove those unnecessary
    features, in order to speed up the evolution.
    """

    def on_obs_processing(self, wrapped_obs):
        wrapped_obs.value = wrapped_obs.value[:-10]


if __name__ == "__main__":
    # NEvoPy provides a general fitness function that works with most gym
    # environments, so we don't have to worry about implementing it. The
    # parameter "default_num_episodes" below is the number of "tries" the agent
    # will have in each playing session.
    fitness_function = ne.utils.GymFitnessFunction(
        make_env=make_env,
        default_num_episodes=10,
        callbacks=[PreProcessObsGymCallback()],
    )

    # Creating a new population of NEAT genomes:
    # (we're going to use the default settings)
    population = ne.neat.population.NeatPopulation(size=200,
                                                   num_inputs=14,
                                                   num_outputs=4,
                                                   config=CONFIG)

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
