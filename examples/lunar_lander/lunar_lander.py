import gym
from nevopy import neat
import time
import numpy as np

env_name = 'LunarLander-v2'

num_episodes = 5
render_fps = 60

num_inputs = 8
num_outputs = 4


def evaluate(genome, eps=num_episodes, visualize=False):
    env = gym.make(env_name)
    total_reward = 0
    for _ in range(eps):
        obs = env.reset()
        if genome is not None:
            genome.reset_activations()

        while True:
            if visualize:
                env.render()
                time.sleep(1 / render_fps)

            if genome is not None:
                action = np.argmax(genome.process(obs))
            else:
                action = env.action_space.sample()

            obs, reward, done, _ = env.step(action)
            total_reward += reward

            if done:
                break
    env.close()
    return total_reward / eps


if __name__ == "__main__":
    pop = neat.population.Population(size=50,
                                     num_inputs=num_inputs,
                                     num_outputs=num_outputs)
    history = pop.evolve(generations=32, fitness_function=evaluate)
    best = pop.fittest()

    print("\nEvaluation (100 episodes):")
    print(f". Random agent score (100 eps): {evaluate(None, eps=100)}")
    print(f". Evolved agent: {evaluate(best, eps=100)}")

    print("\nVisualization:")
    print(f"1) Random agent: {evaluate(None, eps=1, visualize=True)}")
    print(f"2) Evolved agent: {evaluate(best, eps=1, visualize=True)}")

    best.visualize()
    history.visualize(log_scale=False)
