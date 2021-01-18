from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from nevopy import neat
from skimage.transform import resize
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from datetime import datetime
import numpy as np


#################################### CONFIG ####################################
GENERATIONS = 500
POP_SIZE = 100
RENDER_FPS = 70
MAX_STEPS = float("inf")
MAX_IDLE_STEPS = 64
DELTA_X_IDLE_THRESHOLD = 5
LIVES = 1
CONFIG = neat.config.Config(
    new_node_mutation_chance=(0.07, 0.5),
    new_connection_mutation_chance=(0.07, 0.5),
    enable_connection_mutation_chance=(0.07, 0.5),

    disable_inherited_connection_chance=0.75,
    mating_chance=0.75,

    weight_mutation_chance=(0.7, 0.9),
    weight_perturbation_pc=(0.1, 0.5),
    weight_reset_chance=(0.1, 0.32),

    mass_extinction_threshold=32,
    maex_improvement_threshold_pc=0.05,
    species_distance_threshold=1.2,

    infanticide_output_nodes=False,
    infanticide_input_nodes=False,
)
################################################################################


_CACHE_IMGS = False
_SIMPLIFIED_IMGS_CACHE = []


def simplify_img(img, show=False):
    img = rgb2gray(img[75:215, :])
    w, h = img.shape[0] // 9, img.shape[1] // 9
    img = resize(img, (w, h))

    if _CACHE_IMGS:
        _SIMPLIFIED_IMGS_CACHE.append(img)

    if show:
        print(f"Simplified image shape: {img.shape}")
        plt.imshow(img, cmap=plt.cm.gray)
        plt.show()

    return img.flatten()


def video():
    fig = plt.figure()
    img = plt.imshow(_SIMPLIFIED_IMGS_CACHE[0], cmap=plt.cm.gray)

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

    steps = 0
    while steps < max_steps:
        steps += 1
        if visualize:
            env.render()
            time.sleep(1 / RENDER_FPS)

        img = simplify_img(img)
        if genome is not None:
            action = np.argmax(genome.process(img))
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
        num_inputs = len(sample_img)
        print(f"Number of input values: {num_inputs}")
        num_outputs = len(SIMPLE_MOVEMENT)
        print(f"Number of output values: {num_outputs}")

    # evolution
    p = neat.population.Population(size=POP_SIZE,
                                   num_inputs=num_inputs,
                                   num_outputs=num_outputs)
    history = p.evolve(generations=GENERATIONS,
                       fitness_function=evaluate)
    history.visualize(log_scale=False)
    return p


if __name__ == "__main__":
    pop = best = None
    while True:
        try:
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
                    pop = neat.population.Population.load(file_pathname)
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
                f = evaluate(None, visualize=True)
                print(f"Random agent: {f}")
            # exit
            elif key == 0:
                break
            # invalid input
            else:
                raise ValueError()
        except ValueError:
            print("\nInvalid key!")
