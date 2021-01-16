"""
NEvoPY
todo
"""

import nevopy
from timeit import default_timer as timer
import random


# =============== MAKING XOR DATA ==================
num_variables = 2
assert num_variables > 1

xor_inputs = []
xor_outputs = []

for num in range(2 ** num_variables):
    binary = bin(num)[2:].zfill(num_variables)
    xin = [int(binary[0])]
    xout = int(binary[0])
    for bit in binary[1:]:
        xin.append(int(bit))
        xout ^= int(bit)
    xor_inputs.append(xin)
    xor_outputs.append(xout)
# ===================================================


def eval_genome(genome: nevopy.neat.genome.Genome,
                shuffle=True,
                log=False) -> float:
    idx = list(range(len(xor_inputs)))
    if shuffle:
        random.shuffle(idx)

    error = 0
    for i in idx:
        x, y = xor_inputs[i], xor_outputs[i]
        genome.reset_activations()
        h = genome.process(x)[0]
        error += (y - h) ** 2
        if log:
            print(f"IN: {x}  |  OUT: {h}  |  TARGET: {y}")
    if log:
        print(f"\nError: {error}")
    return 1/error


if __name__ == "__main__":
    runs = 1
    total_time = 0
    pop = history = None

    for r in range(runs):
        start_time = timer()
        pop = nevopy.neat.population.Population(
            size=150,
            num_inputs=len(xor_inputs[0]),
            num_outputs=1,
        )
        history = pop.evolve(generations=100,
                             fitness_function=eval_genome,
                             verbose=2)

        deltaT = timer() - start_time
        total_time += deltaT

    best = pop.fittest()
    eval_genome(best, log=True)
    # print(best.info())
    best.visualize()

    print("\n" + 20*"=" +
          f"\nTotal time: {total_time}s"
          f"\nAvg. time: {total_time / runs}s\n")

    history.visualize()

    # print("\n\n\nSaving population...")
    # pop.save("./test/test_pop.pkl")
    #
    # del pop
    # del best
    #
    # pop = nevopy.neat.population.Population.load("./test/test_pop.pkl")
    # print(pop.info())
    # best = pop.fittest()
    # eval_genome(best, log=True)
    # print(best.info())
    # best.visualize()
