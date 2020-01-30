from NSGA import *
from other import *
np.set_printoptions(precision=2)

low_range = -5
high_range = +5
max_generation = 500
population_size = 500
objectives = []


def f1(x):
    return np.power((x - 2), 2) + np.power((x - 1), 2) + 2


def f2(x):
    return 9 * x - np.power((x - 1), 2)


objectives.append(f1)
objectives.append(f2)
# Main part of the implementation
problem1 = NSGA(
    max_generation=max_generation,
    population_size=population_size,
    objectives=objectives,
    crossover_probability=0,
    mutation_probability=0.8,
    initial_population_range=[low_range, high_range],
    sharing_sigma=0.01,
    tournament_q=5,
    parent_selection_size=30,
    pareto_region=[[1, 2], [4, 5]],
    number_of_subregions=10
)

solutions = problem1.run()
