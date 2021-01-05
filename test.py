"""
NEvoPy
todo
"""

import nevopy.neat
from timeit import default_timer as timer


xor_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
xor_outputs = [0, 1, 1, 0]


def eval_genome(genome, log=False):
    genome.reset_activations()
    error = 0
    for x, y in zip(xor_inputs, xor_outputs):
        h = genome.process(x)[0]
        error += (y - h) ** 2
        if log:
            print(f"IN: {x}  |  OUT: {h}  |  TARGET: {y}")
    if log:
        print(f"\nError: {error}")
    return 1/error


if __name__ == "__main__":
    start_time = timer()

    pop = nevopy.neat.population.Population(size=1000, num_inputs=2, num_outputs=1)
    pop.evolve(generations=200, fitness_function=eval_genome)

    deltaT = timer() - start_time

    best = pop.fittest()
    eval_genome(best, log=True)
    best.visualize()

    print(f"\nProcessing time: {deltaT}s\n")

    """id_handler = IdHandler(2 + 1 + 1)
    pop_size = 250
    pop = [nevopy.neat.genome.Genome(num_inputs=2, num_outputs=1, id_handler=id_handler,
                                     config=nevopy.neat.config.Config()) for _ in range(pop_size)]
    for _ in range(100):
        id_handler.reset()

        # eval fitness
        for genome in pop:
            genome.fitness = eval_genome(genome)

        fitness = [g.fitness for g in pop]
        elite = pop[np.argmax(fitness)]
        print(f"Generation: {_}/{200}\n")
        print(f"Best fitness: {elite.fitness}")
        print(f"Mean fitness: {np.mean(fitness)}\n")

        print("Predictions/outputs:")
        eval_genome(elite, log=True)
        print("\n" + "#"*25 + "\n")

        pop = [elite]
        for i in range(pop_size - 1):
            baby = elite.deep_copy()
            if nevopy.utils.chance(0.8):
                baby.mutate_weights()
            if nevopy.utils.chance(0.2):
                baby.add_random_hidden_node()
            if nevopy.utils.chance(0.2):
                baby.add_random_connection()
            if nevopy.utils.chance(0.32):
                baby.enable_random_connection()
            pop.append(baby)

    winner = pop[0]
    print(winner.info())
    eval_genome(winner, log=True)
    winner.visualize()

    config = nevopy.neat.config.Config()
    config.print()

    print("\n\n")
    print(config.weight_mutation_chance)"""


