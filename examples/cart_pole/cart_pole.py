import gym
from nevopy import neat
from nevopy import utils

env_name = 'CartPole-v1'

num_episodes = 5
render_fps = 80

num_inputs = 4
num_outputs = 2


def make_env():
    return gym.make(env_name)


if __name__ == "__main__":
    evaluate = utils.GymEnvFitness(make_env=make_env,
                                   num_episodes=num_episodes,
                                   render_fps=render_fps)
    pop = neat.population.Population(size=100,
                                     num_inputs=num_inputs,
                                     num_outputs=num_outputs)
    history = pop.evolve(generations=50,
                         fitness_function=evaluate,
                         callbacks=[
                             neat.callbacks.FitnessEarlyStopping(
                                 fitness_threshold=500,
                                 min_consecutive_generations=3,
                             )
                         ])
    best = pop.fittest()

    print("\nEvaluation (100 episodes):")
    print(f". Random agent score (100 eps): {evaluate(None, eps=100)}")
    print(f". Evolved agent: {evaluate(best, eps=100)}")

    while True:
        try:
            key = int(input("\nEvolution complete!\n"
                            "   [0] Visualize random agent\n"
                            "   [1] Visualize evolved agent\n"
                            "   [2] Exit\n"
                            "Choose an option: "))
            if key == 0:
                print(f"\nRandom agent: {evaluate(None, eps=1, visualize=True)}")
            elif key == 1:
                print(f"\nEvolved agent: {evaluate(best, eps=1, visualize=True)}")
            elif key == 2:
                break
            else:
                raise ValueError()
        except ValueError:
            print("\nInvalid key!")

    print()
    best.visualize()
    history.visualize(log_scale=False)
