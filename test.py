"""
NEvoPy
todo
"""

import nevopy.neat
from timeit import default_timer as timer


# MAKING XOR DATA
num_variables = 12
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


def eval_genome(genome, log=False):
    genome.reset_activations()
    error = 0
    """indices = list(range(len(xor_inputs)))
    random.shuffle(indices)
    for i in indices:
        x, y = xor_inputs[i], xor_outputs[i]"""
    for x, y in zip(xor_inputs, xor_outputs):
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

    for r in range(runs):
        start_time = timer()
        pop = nevopy.neat.population.Population(size=50,
                                                num_inputs=len(xor_inputs[0]),
                                                num_outputs=1)
        pop.evolve(generations=20, fitness_function=eval_genome)

        deltaT = timer() - start_time
        total_time += deltaT

        #best = pop.fittest()
        #eval_genome(best, log=True)
        #best.visualize()

    print("\n" + 20*"=" +
          f"\nTotal time: {total_time}s"
          f"\nAvg. time: {total_time / runs}s")



