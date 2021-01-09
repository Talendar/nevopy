"""
NEvoPy
todo
"""

import nevopy.neat
from timeit import default_timer as timer


# todo: with decreasing the species threshold is causing fitness to not
#  improve?

# todo: why are so few species being born?

# todo: genome saving and checkpoints

# todo: infanticide / euthanasia (remove individuals with no in
#  connections to output nodes or no out connections from input
#  nodes)

# todo: extinction of species that haven't improved


# =============== MAKING XOR DATA ==================
num_variables = 4
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
    for x, y in zip(xor_inputs, xor_outputs):
        # genome.reset_activations()  # todo: check why this causes fitness = 1
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
    scheduler = nevopy.processing.ray_processing.RayProcessingScheduler()

    pop = None
    for r in range(runs):
        start_time = timer()
        pop = nevopy.neat.population.Population(
            size=100,
            num_inputs=len(xor_inputs[0]),
            num_outputs=1,
            processing_scheduler=scheduler,
        )
        pop.evolve(generations=400,
                   fitness_function=eval_genome)

        deltaT = timer() - start_time
        total_time += deltaT

    best = pop.fittest()
    eval_genome(best, log=True)
    best.visualize()

    print("\n" + 20*"=" +
          f"\nTotal time: {total_time}s"
          f"\nAvg. time: {total_time / runs}s")



