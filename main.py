from NSGA import *
from other import *
np.set_printoptions(precision=2)

low_range = -5
high_range = +5
max_generation = 500
population_size = 500
objectives = []
objectives.append(lambda x: np.power(x, 2))
objectives.append(lambda x: np.power(x - 2, 2))

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
    pareto_region=[[0, 2]],
    number_of_subregions=10
)

solutions = problem1.run()
