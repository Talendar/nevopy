import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from nevopy.fixed_topology import FixedTopologyPopulation
from nevopy.fixed_topology import FixedTopologyGenome
from nevopy.fixed_topology import FixedTopologyConfig
from nevopy.fixed_topology.layers import TFDenseLayer
from nevopy.fixed_topology.layers import mating
from nevopy.processing.ray_processing import RayProcessingScheduler

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from datetime import datetime
import numpy as np


#################################### CONFIG ####################################
GENERATIONS = 300
POP_SIZE = 100
LAYERS = [
    TFDenseLayer(units=64,
                 activation="relu",
                 mating_func=mating.weights_avg_mating,
                 mutable=True),
    TFDenseLayer(units=64,
                 activation="relu",
                 mating_func=mating.weights_avg_mating,
                 mutable=True),
    TFDenseLayer(units=len(SIMPLE_MOVEMENT),
                 activation="relu",
                 mating_func=mating.weights_avg_mating,
                 mutable=True),
]
RENDER_FPS = 60
MAX_STEPS = float("inf")
MAX_IDLE_STEPS = 64
DELTA_X_IDLE_THRESHOLD = 5
LIVES = 1
CONFIG = FixedTopologyConfig(
    # weight mutation
    weight_mutation_chance=(0.7, 0.9),
    weight_perturbation_pc=(0.05, 0.3),
    weight_reset_chance=(0.05, 0.3),
    new_weight_interval=(-1, 1),
    # reproduction
    weak_genomes_removal_pc=0.5,
    mating_chance=0.7,
    mating_mode="exchange_weights_mating",
    rank_prob_dist_coefficient=1.8,
    elitism_count=2,
    predatism_chance=0.15,
    # mass extinction
    mass_extinction_threshold=15,
    maex_improvement_threshold_pc=0.05,
)
################################################################################


_CACHE_IMGS = False
_cached_imgs = []


def video():
    fig = plt.figure()
    img = plt.imshow(_cached_imgs[0])

    def update_fig(j):
        img.set_array(_cached_imgs[j])
        return [img]

    ani = animation.FuncAnimation(fig, update_fig,
                                  frames=len(_cached_imgs),
                                  interval=1000 / RENDER_FPS,
                                  blit=True,
                                  repeat_delay=500)
    plt.show()


def make_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    return JoypadSpace(env, SIMPLE_MOVEMENT)


def evaluate(genome, max_steps=MAX_STEPS, visualize=False):
    total_reward = 0
    env = make_env()
    img = env.reset()

    last_highest_x = 0
    idle_steps = 0
    lives = None
    death_count = 0

    steps = 0
    while steps < max_steps:
        steps += 1
        if visualize:
            env.render()
            time.sleep(1 / RENDER_FPS)

        if genome is not None:
            ram = np.expand_dims(env.unwrapped.ram, axis=0)
            action = np.argmax(genome.process(ram).numpy()[0])
        else:
            action = env.action_space.sample()

        if _CACHE_IMGS:
            _cached_imgs.append(img)

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
        test_env.reset()
        ram = np.expand_dims(test_env.unwrapped.ram, axis=0)
        input_shape = ram.shape
        print(f"Input shape: {input_shape}")
        print(f"Sample input: {ram}")
        print(f"Number of output values: {len(SIMPLE_MOVEMENT)}")

    base_genome = FixedTopologyGenome(
        config=CONFIG,
        layers=LAYERS,
        input_shape=input_shape,
    )

    # evolution
    p = FixedTopologyPopulation(
        size=POP_SIZE,
        base_genome=base_genome,
        config=CONFIG,
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
            # pop.save(file_pathname) todo
            # print(f"Population file saved to: {file_pathname}") todo
        # load pop
        elif key == 2:
            file_pathname = input("Path of the population file to be "
                                  "loaded: ")
            try:
                # pop = FixedTopologyPopulation.load(file_pathname) todo
                best = pop.fittest()
                print("Population loaded!")
            except:
                print("Error while loading the file!")
        # visualize best
        elif key == 3:
            if best is None:
                print("Load a population first!")
            else:
                _cached_imgs = []
                _CACHE_IMGS = True
                f = evaluate(best, visualize=True)
                print(f"Evolved agent: {f}")
                video()
                _CACHE_IMGS = False
        # visualize random
        elif key == 4:
            _cached_imgs = []
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
