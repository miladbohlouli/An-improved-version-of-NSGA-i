from NSGA import *
from other import *
np.set_printoptions(precision=2)

low_range = -5
high_range = +5
max_generation = 500
population_size = 500
objectives = []


def f1(x):
    if x <= 1:
        return -x
    elif 1 < x <= 3:
        return -2 + x
    elif 3 < x <= 4:
        return 4 - x
    elif 4 < x:
        return -4 + x


def f2(x):
    return np.power(x - 5, 2)


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
