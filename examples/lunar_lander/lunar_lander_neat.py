""" In this example, we're going to use NEvoPy's implementation of the NEAT
algorithm to learn how to land a spaceship on the moon!

We're going to use the "LunarLander-v2" gym environment to run the simulations.

Authors:
    Gabriel Nogueira (Talendar)
"""

import gym
import nevopy as ne


def make_env():
    """ Makes a new gym 'LunarLander-v2' environment. """
    return gym.make("LunarLander-v2")


if __name__ == "__main__":
    # NEvoPy provides a general fitness function that works with most gym
    # environments, so we don't have to worry about implementing it. The
    # parameter "default_num_episodes" below is the number of "tries" the agent
    # will have in each playing session.
    fitness_function = ne.utils.GymFitnessFunction(
        make_env=make_env,
        default_num_episodes=5,

        # This renderer will allow us to visualize the activation of the
        # network's neurons in realtime! It generates two videos: one with our
        # genome interacting with the environment and another with our neural
        # network's state during each step of the interaction.
        env_renderer=ne.utils.NeatActivationsGymRenderer(
            fps=30,
            input_visualization_info=[
                ne.neat.NodeVisualizationInfo("Pos X", 0, "diff"),
                ne.neat.NodeVisualizationInfo("Pos Y", 0, "diff"),
                ne.neat.NodeVisualizationInfo("Vel X", 0, "diff"),
                ne.neat.NodeVisualizationInfo("Vel Y", 0, "diff"),
                ne.neat.NodeVisualizationInfo("Angle", 0, "diff"),
                ne.neat.NodeVisualizationInfo("Ang vel", 0, "diff"),
                ne.neat.NodeVisualizationInfo("Left leg", 1, "equal"),
                ne.neat.NodeVisualizationInfo("Right leg", 1, "equal"),
            ],
            show_input_values=True,
            output_visualization_info=["NOp", "Left engine", "Down engine",
                                       "Right engine"],
            show_output_values=False,
            output_activate_greatest_only=True,
        ),
    )

    # Creating a new population of NEAT genomes:
    # (we're going to use the default settings)
    population = ne.neat.population.NeatPopulation(size=100,
                                                   num_inputs=8,
                                                   num_outputs=4)

    # We can tell NEvoPy to halt the evolutionary process when a certain fitness
    # value has been achieved by the population. For that, we use a specific
    # type of callback.
    early_stopping_cb = ne.callbacks.FitnessEarlyStopping(
        fitness_threshold=200,
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
