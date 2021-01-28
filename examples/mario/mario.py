import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from nevopy import neat
from nevopy import fixed_topology
from nevopy.fixed_topology.layers import TFConv2DLayer
from nevopy.fixed_topology.layers import TFDenseLayer
from nevopy.fixed_topology.layers import TFFlattenLayer
from nevopy.fixed_topology.layers import mating
from nevopy.processing.ray_processing import RayProcessingScheduler

from skimage.transform import resize
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from datetime import datetime
import numpy as np
import tensorflow as tf


#################################### CONFIG ####################################
GENERATIONS = 100
POP_SIZE = 100
IMG_BATCH_SIZE = 4
FITO_LAYERS = [
    TFConv2DLayer(filters=32,
                  kernel_size=(3, 3),
                  strides=(1, 1),
                  mating_func=mating.weights_avg_mating,
                  mutable=True),
    TFConv2DLayer(filters=32,
                  kernel_size=(3, 3),
                  strides=(1, 1),
                  mating_func=mating.weights_avg_mating,
                  mutable=True),
    TFFlattenLayer(),
    TFDenseLayer(units=32,
                 activation="relu",
                 mating_func=mating.weights_avg_mating,
                 mutable=True),
    TFDenseLayer(units=3,
                 activation="relu",
                 mating_func=mating.weights_avg_mating,
                 mutable=True),
]
RENDER_FPS = 60
MAX_STEPS = float("inf")
MAX_IDLE_STEPS = 64
DELTA_X_IDLE_THRESHOLD = 5
LIVES = 1
NEAT_CONFIG = neat.config.NeatConfig(
    weak_genomes_removal_pc=0.7,
    new_node_mutation_chance=(0.1, 0.7),
    new_connection_mutation_chance=(0.1, 0.5),
    enable_connection_mutation_chance=(0.1, 0.5),

    disable_inherited_connection_chance=0.75,
    mating_chance=0.75,

    weight_mutation_chance=(0.7, 0.9),
    weight_perturbation_pc=(0.1, 0.5),
    weight_reset_chance=(0.1, 0.4),

    mass_extinction_threshold=20,
    maex_improvement_threshold_pc=0.05,

    excess_genes_coefficient=2,
    disjoint_genes_coefficient=2,
    weight_difference_coefficient=1.5,
    species_distance_threshold=5,

    infanticide_output_nodes=False,
    infanticide_input_nodes=True,
)
FITO_CONFIG = fixed_topology.config.FixedTopologyConfig(
    weight_mutation_chance=(0.3, 0.5),
    weight_perturbation_pc=(0.05, 0.2),
    weight_reset_chance=(0.01, 0.1),
    new_weight_interval=(-1, 1),

    mating_mode="exchange_weights_mating",
    mass_extinction_threshold=20,
)
################################################################################


_CACHE_IMGS = False
_SIMPLIFIED_IMGS_CACHE = []


def simplify_img(img, show=False):
    # img = rgb2gray(img[70:225, :])
    img = img[70:235]
    w, h = img.shape[0] // 3, img.shape[1] // 3
    img = resize(img, (w, h))

    if _CACHE_IMGS:
        _SIMPLIFIED_IMGS_CACHE.append(img)

    if show:
        print(f"Simplified image shape: {img.shape}")
        plt.imshow(img)
        plt.show()

    img = img.astype(np.float32)
    return img


def video():
    # for img in _SIMPLIFIED_IMGS_CACHE:
    #     plt.imshow(img)
    #     plt.show()

    fig = plt.figure()
    img = plt.imshow(_SIMPLIFIED_IMGS_CACHE[0])

    def update_fig(j):
        img.set_array(_SIMPLIFIED_IMGS_CACHE[j])
        return [img]

    ani = animation.FuncAnimation(fig, update_fig,
                                  frames=len(_SIMPLIFIED_IMGS_CACHE),
                                  interval=1000 / RENDER_FPS,
                                  blit=True,
                                  repeat_delay=500)
    plt.show()


def make_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v2')
    return JoypadSpace(env, SIMPLE_MOVEMENT)


def evaluate(genome, max_steps=MAX_STEPS, visualize=False):
    total_reward = 0
    env = make_env()

    img = env.reset()
    if genome is not None:
        genome.reset_activations()

    last_highest_x = 0
    idle_steps = 0
    lives = None
    death_count = 0

    img_batch = []
    steps = 0
    action = env.action_space.sample()
    while steps < max_steps:
        steps += 1
        if visualize:
            env.render()
            time.sleep(1 / RENDER_FPS)

        img = simplify_img(img)
        if genome is not None:
            img_batch.append(img)
            if len(img_batch) >= IMG_BATCH_SIZE:
                img_stack = np.array(img_batch)
                action = np.argmax(genome.process(img_stack))
                img_batch = []
        else:
            action = env.action_space.sample()

        img, reward, done, info = env.step(action)
        total_reward += reward

        if lives is None:
            lives = info["life"]

        # checking idle
        x_pos, got_flag = info["x_pos"], info["flag_get"]
        dead = lives > info["life"]
        lives = info["life"]

        if got_flag or dead:
            last_highest_x = 0
            idle_steps = 0
        else:
            if (x_pos - last_highest_x) >= DELTA_X_IDLE_THRESHOLD:
                last_highest_x = x_pos
                idle_steps = 0
            else:
                idle_steps += 1

        if dead:
            death_count += 1

        if done or idle_steps > MAX_IDLE_STEPS or death_count >= LIVES:
            break

    env.close()
    return total_reward


def new_pop():
    # preparing
    with make_env() as test_env:
        obs = test_env.reset()
        print("Visualizing sample input image...")
        sample_img = simplify_img(obs, show=True)
        # num_inputs = len(sample_img)
        # print(f"Number of input values: {num_inputs}")
        num_outputs = len(SIMPLE_MOVEMENT)
        print(f"Number of output values: {num_outputs}")

    fito_genome = fixed_topology.genomes.FixedTopologyGenome(
        config=FITO_CONFIG,
        layers=FITO_LAYERS,
    )

    sample_input = tf.random.uniform(shape=[IMG_BATCH_SIZE, *sample_img.shape])
    test_output = fito_genome(sample_input)
    print(f"CNN input shape: {sample_input.shape}")
    print(f"CNN output shape: {test_output.shape} ", end="")
    test_output = tf.reshape(test_output, [-1])
    print(f"(flat: {len(test_output)})")

    base_genome = neat.genomes.FixTopNeatGenome(
        fito_genome=fito_genome,
        num_neat_inputs=len(test_output),
        num_neat_outputs=len(SIMPLE_MOVEMENT),
        config=NEAT_CONFIG,
    )

    # evolution
    p = neat.population.NeatPopulation(
        size=POP_SIZE,
        base_genome=base_genome,
        config=NEAT_CONFIG,
        processing_scheduler=RayProcessingScheduler(num_gpus=0,
                                                    worker_gpu_frac=0),
    )
    history = p.evolve(generations=GENERATIONS,
                       fitness_function=evaluate)
    history.visualize(log_scale=False)
    return p


if __name__ == "__main__":
    # start
    pop = best = None
    while True:
        key = int(input("\n\n< Learning Mario with NEvoPY >\n"
                        "   [1] New population\n"
                        "   [2] Load population\n"  
                        "   [3] Visualize best agent\n"
                        "   [4] Visualize random agent\n"
                        "   [0] Exit\n"
                        "Choose an option: "))
        print("\n")

        # new pop
        if key == 1:
            pop = new_pop()
            best = pop.fittest()
            file_pathname = (
                f"./saved_pop/pop_fit{best.fitness}_"
                f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
            )
            pop.save(file_pathname)
            print(f"Population file saved to: {file_pathname}")
        # load pop
        elif key == 2:
            file_pathname = input("Path of the population file to be "
                                  "loaded: ")
            try:
                pop = neat.population.NeatPopulation.load(file_pathname)
                best = pop.fittest()
                print("Population loaded!")
            except:
                print("Error while loading the file!")
        # visualize best
        elif key == 3:
            if best is None:
                print("Load a population first!")
            else:
                _SIMPLIFIED_IMGS_CACHE = []
                _CACHE_IMGS = True
                f = evaluate(best, visualize=True)
                print(f"Evolved agent: {f}")
                video()
                _CACHE_IMGS = False
        # visualize random
        elif key == 4:
            _SIMPLIFIED_IMGS_CACHE = []
            _CACHE_IMGS = True
            f = evaluate(None, visualize=True)
            print(f"Random agent: {f}")
            video()
            _CACHE_IMGS = False
        # exit
        elif key == 0:
            break
        # invalid input
        else:
            print("\nInvalid key!")
