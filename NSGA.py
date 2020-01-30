import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')
from other import *
import random
import imageio
sns.set(color_codes=True)


class NSGA:
    def __init__(self,
                 population_size=100,
                 initial_population_range=None,
                 max_generation=100,
                 objectives=[],
                 crossover_probability=0.5,
                 mutation_probability = 0.5,
                 parent_selection_size = 100,
                 sharing_sigma = 0.1,
                 tournament_q = 5,
                 number_of_subregions=10,
                 pareto_region=[]):
        """
        The constructor of the class
        :param population_size: The size of the population(mu)
        :param initial_population_range: The range of the initial population
        :param max_generation: the maximum number of generations
        :param objectives: A list containing the objectives
        :param crossover_probability: the probability of applying crossover
        :param mutation_probability: the probability of applying mutation
        :param parent_selection_size: the size of parent selection
        :param sharing_sigma: the region of where two solutions are classified as neighbours
        :param tournament_q: the selection q used in q tournament
        :param number_of_subregions: The number of subregions used in plotting the distribution of
        the generations on pareto region
        :param pareto_region: A list containing the tuple of pareto regions
        """

        # Defining the parameters
        if initial_population_range is None:
            self.__initial_population_range = [-3, 5]
        else:
            self.__initial_population_range = initial_population_range
        self._tournament_q = tournament_q
        self._population_size = population_size
        self._max_generation = max_generation
        self._objectives = objectives
        self._parent_selection_size = parent_selection_size
        self._sharing_sigma = sharing_sigma
        self._crossover_probability = crossover_probability
        self._mutation_probability = mutation_probability
        self._number_of_sub_regions = number_of_subregions
        self._pareto_region = pareto_region
        self._population_plot_images = []

        assert self._parent_selection_size <= self._population_size

        self.__population = []
        self.__generation_counter = 0
        self.__subregions = NSGA.__cal_sub_regions(self._pareto_region, self._number_of_sub_regions)

        # Initiating the population
        self.__initiate_population()

        # calculating the values for the objectives
        self.__distribution_matrix = np.zeros((max_generation, number_of_subregions))
        self.__values_for_objectives = np.zeros((population_size, len(self._objectives)))
        self.__fronts_per_generation = np.zeros(self._max_generation)
        self.__selection_pressure_per_generation = np.zeros(max_generation)

    def run(self):
        """
        The main function used for running the algorithm which handles all the operations
        :return self.__population: Defines the population after _max_generation number of iterations
        """
        for _ in range(self._max_generation):

            for i, chromosome in enumerate(self.__population):
                for j, objective in enumerate(self._objectives):
                    self.__values_for_objectives[i][j] = objective(chromosome)

            print("\n********************************************\n"
                  "processing generation %d out of %d\n"
                  "********************************************"
                  %(self.__generation_counter + 1, self._max_generation))
            fitness_values = self.__dummy_fitness_with_sharing()
            selected_parents = self.__parent_selection(fitness_values)
            children = self.__reproduction(selected_parents)
            self.__next_generation_selection(children, fitness_values)
            self.__selection_pressure_per_generation[self.__generation_counter] = \
                (np.max(fitness_values) / np.sum(fitness_values) * self._population_size)

            for i in range(self._number_of_sub_regions):
                self.__distribution_matrix[self.__generation_counter, i] = \
                    np.sum(np.logical_and(self.__population >= self.__subregions[i][0], self.__population < self.__subregions[i][1]))

            self.__generation_counter += 1

        # Used for plotting
        plt.figure(figsize=(13, 7))
        plt.subplots_adjust(hspace=0.5, wspace=0.3)
        plt.subplot(2, 2, 1)
        self.plot_val_generation(self.__fronts_per_generation, "number of fronts")

        plt.subplot(2, 2, 2)
        self.plot_distribution()

        plt.subplot(2, 2, 3)
        self.plot_val_generation(self.__selection_pressure_per_generation, "Selection pressure")

        plt.subplot(2, 2, 4)
        plt.title("The final solutions after %d generations"%(self._max_generation), size=10)
        plt.xlabel("f1", size=7), plt.ylabel("f2", size=7)
        plot(self._objectives,
             low_range=np.min(self.__population) - 1,
             high_range=np.max(self.__population) + 1)

        scatter(self.__population,
                self._objectives)

        plt.show()

        return self.__population

    @staticmethod
    def __cal_sub_regions(pareto_regions, bin_count):
        """
        Is used to calculate the subregions of the pareto region, to be used in distribution plotting
        :param pareto_regions: A list containing the pareto regions list in [start, end] format. Note that
        the length of subregions could vary and is not constant
        :param bin_count: the number of the subregions we would like to extract from the input pareto_regions
        :return: a python list containing the lists of the subregions in format [start, end]
        """
        # The slotted subregions of the pareto region with the length of bin_count
        sub_regions_list = []

        # The total length of the pareto regions
        total_range = 0
        for i, sub_region in enumerate(pareto_regions):
            pareto_regions[i][0] = round(float(pareto_regions[i][0]), 1)
            pareto_regions[i][1] = round(float(pareto_regions[i][1]), 1)
            total_range += sub_region[1] - sub_region[0]

        # length of each sub_region according to the bin_count and pareto_regions length
        sub_region_length = round((total_range / bin_count), 1)
        for i in range(len(pareto_regions)):
            decision_boundary = pareto_regions[i][0]
            while True:
                if decision_boundary >= pareto_regions[i][1]:
                    break
                sub_regions_list.append([decision_boundary, round(decision_boundary + sub_region_length, 1)])
                decision_boundary += sub_region_length
                decision_boundary = round(decision_boundary, 1)

        # Check weather the length of the calculated subregions is equal to the bin_count
        assert len(sub_regions_list) == bin_count
        return sub_regions_list

    def plot_distribution(self):
        """
        This is used for plotting the distribution of the chromosomes.
        :return: void
        """
        for i in range(self._number_of_sub_regions):
            plt.title("Distribution of the solutions on the pareto", size=10)
            plt.xlabel("Generation", size=7), plt.ylabel("The variation", size=7)
            plt.plot(range(self._max_generation), self.__distribution_matrix[:, i], linewidth=1)

    def plot_val_generation(self, values, eval_metric):
        """
        This function plots the given values with respect to generation, one example could be selection pressure
        :param values: An array with the same length as the max_generation containing the values
        :param eval_metric: The name of the eval_metric in string format
        :return: void
        """
        plt.title(str(eval_metric) + " changes with respect to generation", size=10)
        plt.xlabel("Generation", size=7), plt.ylabel(eval_metric, size=7)
        plt.plot(range(self._max_generation), values, color='red', linewidth=1)

    def __initiate_population(self):
        """
        Function is used for initiating the first population, The initial population is generated
        with a uniform distribution. ----> done
        :param population_range: The range of the initial population
        :return: void
        """

        self.__population = np.random.uniform(self.__initial_population_range[0],
                                              self.__initial_population_range[1],
                                              self._population_size)

    def __parent_selection(self, fitness_values):
        """
        Used for selecting n number of the parents ----> done
        :param probs: The fitness values that we have calculated
        :return: the selected parents
        """
        return NSGA.__tournament(self.__population,
                                 fitness_values,
                                 self._parent_selection_size,
                                 self._tournament_q)

    def __reproduction(self, parents):
        """
        The following function handles these operations: ----> done
            1. reproducing the new children (cross over)
            2. mutating the generated children

        :param parents: The population of the parnets we would like to choose from
        :return: the produced children
        """
        children = []
        np.random.shuffle(parents)
        for i in range(0, len(parents - 1), 2):
            parent1, parent2 = parents[i], parents[i + 1]
            child1, child2 = NSGA.__cross_over(parent1, parent2, parameters={"prob":self._crossover_probability})
            child1 = NSGA.__mutation(child1, self._mutation_probability)
            child2 = NSGA.__mutation(child2, self._mutation_probability)
            children.append(float(child1))
            children.append(float(child2))
        np.random.shuffle(children)
        return children[:self._parent_selection_size]

    def __dummy_fitness_with_sharing(self):
        """
        The following operations are handled in the below function:
            1. Classifying the population according to their domination ---> done
            2. Assigning
        :return: list of the fitness values for selection
        """

        fitness_values = np.zeros(self._population_size)
        classified_chromosomes = self.fast_non_dominated_sorting()

        num_fronts = len(classified_chromosomes)
        print("num fronts: %d"%(num_fronts))
        front_shares = []
        self.__fronts_per_generation[self.__generation_counter] = num_fronts
        for i in range(num_fronts):
            front_shares.append(1 / self.sharing_value_calculation(classified_chromosomes[i]))

        for i in range(num_fronts-2, -1, -1):
            front_shares[i] += max(front_shares[i+1])

        for i in range(num_fronts):
            for j, item in enumerate(classified_chromosomes[i]):
                fitness_values[item] = front_shares[i][j]

        return fitness_values

    def sharing_value_calculation(self,
                                  front):
        """
        This function gets a front as input, finds the sharing value of the front which represents how dense the
        chromosomes in the front are ----> done
        :param front: The list of the chromosomes we are interested in selecting from
        :return : A list with the same length of the input front but with float values indicating the share parameter
        for each solution
        """
        sharing_values = np.zeros(len(front))
        for i, item in enumerate(front):
            distances = np.abs(self.__population[front] - self.__population[item])
            temp_dist = np.copy(distances)
            distances[temp_dist > self._sharing_sigma] = 0
            distances[temp_dist < self._sharing_sigma] = 1 - (distances[temp_dist < self._sharing_sigma] / self._sharing_sigma)
            sharing_values[i] = np.sum(distances)

        return sharing_values

    def fast_non_dominated_sorting(self):
        """
        We will try to find the non_dominated solution of the chromosome with presumption that the optimization problem is
        a minimization problem, We should have this in mind that this has been inspired from NSGA-ii since the domination
        method in NSGA-i was not quite well. ----> done
        :return :a list containing all the classes of the chromosomes
        """
        fronts = []
        chromosomes = np.copy(self.__population)

        num_chromosomes = len(chromosomes)
        Np = np.zeros(num_chromosomes)
        Sp = []

        for i in range(num_chromosomes):
            temp_sp = []
            for j in range(num_chromosomes):
                if not i == j:
                    if np.all(self.__values_for_objectives[i, :] < self.__values_for_objectives[j, :]):
                        Np[j] += 1
                        temp_sp.append(j)
            Sp.append(temp_sp)

        Q = []
        for i in range(num_chromosomes):
            if Np[i] == 0:
                Q.append(i)
        fronts.append(Q)

        i = 0
        traversed_items = len(Q)
        while not traversed_items == num_chromosomes:
            temp_fronts = []
            for item in Q:
                for dominated in Sp[item]:
                    Np[dominated] -= 1
                    if Np[dominated] == 0:
                        temp_fronts.append(dominated)

            Q = np.copy(temp_fronts)
            fronts.append(temp_fronts)
            traversed_items += len(temp_fronts)
        return fronts

    def __next_generation_selection(self,
                                    children,
                                    fitness_values):
        """
        This function handles gathering the parents and children and assigning the fitness values finally the
        next generation will be selected with tournament selection method ----> done

        :param children: the generated children in list format
        :param parents: the parents
        """
        num_children = len(children)
        self.__population[np.argsort(fitness_values)[:num_children]] = np.array(children)


    @staticmethod
    def __mutation(chromosome, prob):
        """
        Is for mutation in order expand the exploration region, for this project the mutation has been put away
        since the objectives are convex  ----> done

        :param chromosome: the generated children in list format
        :return parents: the parents
        """
        if np.random.random() < prob:
            chromosome = chromosome + np.random.normal(0, 0.01, 1)
        return chromosome

    @staticmethod
    def __cross_over(chromosome1,
                     chromosome2,
                     parameters):
        """
        This is used for crossing over the given inputs and building the child ---> done

        :param chromosome1, chromosome2: the parents of the child
        :param parameters: The possible required parameters for cross over
        """

        prob = parameters['prob']
        if np.random.random() < prob:
            prob = np.random.random()
            child1 = (prob * chromosome1 + (1 - prob) * chromosome2)

            prob = np.random.random()
            child2 = (prob * chromosome1 + (1 - prob) * chromosome2)

        else:
            child1, child2 = chromosome1, chromosome2
        return child1, child2

    @staticmethod
    def __tournament(items,
                     probabilities,
                     n,
                     q=10):
        """
        The tournament method for selection ----> done

        :param items:  Items that want to choose from them, np.array or list
        :param probabilities:  Probabilities of each item, in fact the fitness values of the items
        :param q: The number of the items wwe will choose on each iteration, integer
        :param n: number of selected item(s), Integer
        :return:
        """

        # checking the integrity of the inputs
        assert n >= q
        assert len(items) == len(probabilities)
        assert q != 0
        probabilities = np.array(probabilities)

        items = np.array(items)
        if n == 0:
            return np.array([])

        else:
            index = np.arange(len(items))
            np.random.shuffle(index)
            items = items[index]
            probabilities = probabilities[index]

            selected_items = []
            len_items = len(items)

            for i in range(n):
                indexes = np.random.choice(np.arange(len_items), q, replace=False)
                selected_items.append(items[indexes[np.argmax(probabilities[indexes])]])
        return np.array(selected_items)

    # The following parts of the class have been depreciated and are not currently being used
    @staticmethod
    def save_as_gif(images_list):
        # index = np.random.randint(0, len(images_list), 1)
        index = random.randint(0, len(images_list))
        print(index)
        plt.imshow(images_list[index])
        plt.show()

    def plot_convert_image(self):
        """
        This function plots the current population according to the objectives and converts it to image
        :return: The image format of the plot with the population and the objectives
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        for objective in self._objectives:
            plot(ax, objective, low_range=self.__initial_population_range[0],
                 high_range=self.__initial_population_range[1])
            scatter(ax, self.__population, objective)

        ax.set_ylim([0, 8])
        ax.set_xlim(self.__initial_population_range[0], self.__initial_population_range[1])
        ax.set(xlabel="Generation {}".format(self.__generation_counter))

        fig.canvas.draw()  # draw the canvas, cache the renderer
        image = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return image

