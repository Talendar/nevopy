import gym
from nevopy import neat
import time

env_name = 'CartPole-v1'
num_episodes = 5
render_fps = 80


def evaluate(genome, visualize=False):
    env = gym.make(env_name)
    total_reward = 0
    sess = num_episodes if not visualize else 1
    for _ in range(sess):
        obs = env.reset()
        genome.reset_activations()
        while True:
            if visualize:
                env.render()
                time.sleep(1 / render_fps)
            action = round(genome.process(obs)[0])
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
    env.close()
    return total_reward / sess


def random_agent():
    env = gym.make(env_name)
    env.reset()
    total_reward = 0
    while True:
        env.render()
        time.sleep(1 / render_fps)
        action = env.action_space.sample()
        _, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    env.close()
    return total_reward


if __name__ == "__main__":
    pop = neat.population.Population(size=50,
                                     num_inputs=4,
                                     num_outputs=1)
    pop.evolve(generations=10, fitness_function=evaluate)
    best = pop.fittest()
    print(f"\nRandom agent: {random_agent()}")
    print(f"Evolved agent: {evaluate(best, visualize=True)}")
