"""
NEvoPy
todo
"""

import nevopy.neat
from timeit import default_timer as timer
import random

# import ray
# ray.init()


xor_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
xor_outputs = [0, 1, 1, 0]


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
    """input_values = range(100000)

    # no parallelization
    start_time = timer()
    results = [f(input_values) for _ in range(100)]
    print(f"Non-ray time: {1000 * (timer() - start_time) : .5f} ms")
    print(results)
    print(f"min = {np.min(results)}  |  max = {np.max(results)}")

    # ray parallelization
    start_time = timer()
    results = ray.get([f_ray.remote(input_values) for _ in range(100)])
    print(f"\nRay time: {1000 * (timer() - start_time) : .5f} ms")
    print(results)
    print(f"min = {np.min(results)}  |  max = {np.max(results)}")"""

    runs = 5
    total_time = 0

    for r in range(runs):
        start_time = timer()
        pop = nevopy.neat.population.Population(size=1000, num_inputs=2,
                                                num_outputs=1)
        pop.evolve(generations=32, fitness_function=eval_genome)

        deltaT = timer() - start_time
        total_time += deltaT

        #best = pop.fittest()
        #eval_genome(best, log=True)
        #best.visualize()

    print("\n" + 20*"=" +
          f"\nTotal time: {total_time}s"
          f"\nAvg. time: {total_time / runs}s")



