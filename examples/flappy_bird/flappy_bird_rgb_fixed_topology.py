""" In this example, we're going to use NEvoPy's implementation of a classic
fixed-topology neuroevolution algorithm to learn how to play Flappy Bird!

We're going to use the "FlappyBird-v0" gym environment (available here:
https://github.com/Talendar/flappy-bird-gym) to simulate the game.

Authors:
    Gabriel Nogueira (Talendar)
"""

import os

import flappy_bird_gym
import matplotlib.pyplot as plt
import nevopy as ne
import numpy as np
import skimage.color
import skimage.transform


# In some scenarios, using a GPU with TensorFlow during neuroevolution might
# slow things down considerably. Since in this example we're dealing with small
# images in batches of 1 item, it's not worth using the GPU. The line below
# makes TensorFlow to use only CPUs.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Settings for the ray processing scheduler.
RAY_CONFIG = dict(
    num_gpus=0,
    worker_gpu_frac=0,
)

# Defining a base genome for our population. It will serve as a model for the
# other genomes of the population, which will have the same topology as it.
BASE_GENOME = ne.fixed_topology.FixedTopologyGenome(

    # List with the layers of the base genome. Other genomes in the population
    # will have similar layers (with the same topology).
    layers=[
        ne.fixed_topology.layers.TFConv2DLayer(
            filters=8,
            kernel_size=(10, 10),
            strides=(5, 5),
            mating_func=ne.fixed_topology.layers.mating.exchange_units_mating,
        ),
        ne.fixed_topology.layers.TFFlattenLayer(),
        ne.fixed_topology.layers.TFDenseLayer(32, activation="relu"),
        ne.fixed_topology.layers.TFDenseLayer(1, activation="sigmoid")
    ],
)


def pre_process_obs(img, env=None, show=False):
    img = skimage.color.rgb2gray(img)
    img = np.moveaxis(img, source=1, destination=0)  # width <-> height
    img = img[10:-90]

    w, h = img.shape[0] // 6, img.shape[1] // 6
    img = skimage.transform.resize(img, (w, h))

    if show:
        print(f"Simplified image shape: {img.shape}")
        plt.imshow(img, cmap='gray')
        plt.show()

    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    return img


def make_env():
    """ Makes a new gym 'FlappyBird-rgb-v0' environment. """
    return flappy_bird_gym.make("FlappyBird-rgb-v0")


if __name__ == "__main__":
    # Let's start by visualizing some stuff.
    with make_env() as test_env:
        obs = test_env.reset()
        print("Visualizing sample input image...")
        sample_input = pre_process_obs(obs, show=True)

        # Feeding the sample image to our base genome, so it can register
        # the shape of the inputs:
        print(f"\nSample input shape: {sample_input.shape}")
        print(f"Sample output: {BASE_GENOME(sample_input)}")

        print("\nVisualizing base genome's topology...")
        BASE_GENOME.visualize()

    # NEvoPy provides a general fitness function that works with most gym
    # environments, so we don't have to worry about implementing it.
    fitness_function = ne.utils.GymEnvFitness(
        make_env=make_env,

        # Number of "tries" the agent will have in each playing session:
        num_episodes=5,

        # Let's pass the pre_process_obs function we implemented earlier as
        # argument, so we can pre-process the images from the game screen
        # before feeding them to our genomes.
        pre_process_obs=pre_process_obs,
    )

    # Creating a new random population of fixed-topology genomes:
    # (we're going to use the default settings)
    population = ne.genetic_algorithm.GeneticPopulation(
        size=64,
        base_genome=BASE_GENOME,
        processing_scheduler=ne.processing.RayProcessingScheduler(**RAY_CONFIG),
    )

    # We can tell NEvoPy to halt the evolutionary process when a certain fitness
    # value has been achieved by the population. For that, we use a specific
    # type of callback.
    early_stopping_cb = ne.callbacks.FitnessEarlyStopping(
        fitness_threshold=5000,
        min_consecutive_generations=5,
    )

    # Evolving the population (this might take some minutes):
    history = population.evolve(generations=100,
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
    print(f". Random genome score: {fitness_function(None, eps=25)}")
    print(f". Evolved genome: {fitness_function(best_genome, eps=25)}")

    # Visualizing our best genome and a random agent playing:
    while True:
        key = int(input("\nEvolution complete!\n"
                        "   [0] Visualize random genome\n"
                        "   [1] Visualize evolved genome\n"
                        "   [2] Exit\n"
                        "Choose an option: "))
        if key == 0:
            print(f"\nRandom genome: "
                  f"{fitness_function(None, eps=1, visualize=True)}")
        elif key == 1:
            print(f"\nEvolved genome: "
                  f"{fitness_function(best_genome, eps=1, visualize=True)}")
        elif key == 2:
            break
        else:
            print("\nInvalid key!")
