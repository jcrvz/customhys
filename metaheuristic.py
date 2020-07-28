# -*- coding: utf-8 -*-
"""
This module contains the Metaheuristic class.

Created on Thu Sep 26 16:56:01 2019

@author: Jorge Mario Cruz-Duarte (jcrvz.github.io), e-mail: jorge.cruz@tec.mx
"""

import numpy as np
from population import Population
import operators as Operators

__all__ = ['Metaheuristic', 'Population', 'Operators']
__operators__ = Operators.__all__
__selectors__ = ['greedy', 'probabilistic', 'metropolis', 'all', 'none']


class Metaheuristic:
    """
        This is the Metaheuristic class, each object corresponds to a metaheuristic implemented with a sequence of
        search operators from Operators, and it is based on a population from Population.
    """
    def __init__(self, problem, search_operators, num_agents=30, num_iterations=100):
        """
        Create a population-based metaheuristic by employing different simple search operators.

        :param dict problem:
            This is a dictionary containing the 'function' that maps a 1-by-D array of real values ​​to a real value,
            'is_constrained' flag that indicates the solution is inside the search space, and the 'boundaries' (a tuple
            with two lists of size D). These two lists correspond to the lower and upper limits of domain, such as:
            ``boundaries = (lower_boundaries, upper_boundaries)``

            **Note:** Dimensions (D) of search domain are read from these boundaries. The problem can be obtained from
            the ``benchmark_func`` module.
        :param list search_operators:
            A list of available search operators. These operators must correspond to those available in the
            ``operators`` module.
        :param int num_agents: Optional.
            Number of agents or population size. The default is 30.
        :param int num_iterations: Optional.
            Number of iterations or generations that the metaheuristic is going to perform. The default is 100.

        :return: None.
        """
        # Read the problem function
        self.problem_function = problem['function']

        # Create population
        self.pop = Population(problem['boundaries'], num_agents, problem['is_constrained'])

        # Check and read the search_operators
        self.operators, self.selectors = Operators.process_operators(search_operators)

        # Define the maximum number of iterations
        self.num_iterations = num_iterations

        # Read the number of dimensions
        self.num_dimensions = self.pop.num_dimensions

        # Read the number of agents
        self.num_agents = num_agents

        # Initialise historical variables
        self.historical = dict()

        # Set additional variables
        self.verbose = False

    def run(self):
        """
        Run the metaheuristic for solving the defined problem.

        :return: None.

        """
        # Set initial iteration
        self.pop.iteration = 0

        # Initialise the population
        self.pop.initialise_positions()  # Default: random

        # Evaluate fitness values
        self.pop.evaluate_fitness(self.problem_function)

        # Update population, particular, and global
        self.pop.update_positions('population', 'all')  # Default
        self.pop.update_positions('particular', 'all')
        self.pop.update_positions('global', 'greedy')

        # Initialise and update historical variables
        self._reset_historicals()
        self._update_historicals()

        # Report which operators are going to use
        self._verbose('\nSearch operators to employ:')
        for operator, selector in zip(self.operators, self.selectors):
            self._verbose("{} with {}".format(operator, selector))
        self._verbose("{}".format('-' * 50))

        # Start optimisaton procedure
        for iteration in range(1, self.num_iterations + 1):
            # Update the current iteration
            self.pop.iteration = iteration

            # Implement the sequence of operators and selectors
            for operator, selector in zip(self.operators, self.selectors):
                # Split operator
                operator_name, operator_params = operator.split('(')

                # Apply an operator
                exec('Operators.' + operator_name + '(self.pop,' + operator_params)

                # Evaluate fitness values
                self.pop.evaluate_fitness(self.problem_function)

                # Update population
                if selector in __selectors__:
                    self.pop.update_positions('population', selector)
                else:
                    self.pop.update_positions()

                # Update global position
                self.pop.update_positions('global', 'greedy')

            # Update historical variables
            self._update_historicals()

            # Verbose (if so) some information
            self._verbose('{}\npop. radius: {}'.format(iteration, self.historical['radius'][-1]))
            self._verbose(self.pop.get_state())

    def get_solution(self):
        """
        Deliver the last position and fitness value obtained after ``run`` the metaheuristic procedure.

        :returns: ndarray, float
        """
        return self.historical['position'][-1], self.historical['fitness'][-1]

    # TODO: Integrate this property again
    # @property
    # Deprecated!
    # def show_performance(self):
    #     """
    #     Show the solution evolution during the iterative process.
    #
    #     Returns
    #     -------
    #     None.
    #
    #     """
    #     # Show historical fitness
    #     fig1, ax1 = plt.subplots()
    #
    #     color = 'tab:red'
    #     plt.xlabel("Iterations")
    #     ax1.set_ylabel("Global Fitness", color=color)
    #     ax1.plot(np.arange(0, self.num_iterations + 1),
    #              self.historical['fitness'], color=color)
    #     ax1.tick_params(axis='y', labelcolor=color)
    #     ax1.set_yscale('linear')
    #
    #     ax2 = ax1.twinx()
    #
    #     color = 'tab:blue'
    #     ax2.set_ylabel('Population radius', color=color)
    #     ax2.plot(np.arange(0, self.num_iterations + 1),
    #              self.historical['radius'], color=color)
    #     ax2.tick_params(axis='y', labelcolor=color)
    #     ax2.set_yscale('log')
    #
    #     fig1.tight_layout()
    #     plt.show()

    def _reset_historicals(self):
        """
        Reset the ``historical`` variables.

        :return: None.
        """
        self.historical = dict(fitness=list(), position=list(), centroid=list(), radius=list())

    def _update_historicals(self):
        """
        Update the ``historical`` variables.

        :return: None.
        """
        # Update historical variables
        self.historical['fitness'].append(np.copy(self.pop.global_best_fitness))
        self.historical['position'].append(np.copy(self.pop.global_best_position))

        # Update population centroid and radius
        current_centroid = np.array(self.pop.positions).mean(0)
        self.historical['centroid'].append(np.copy(current_centroid))
        self.historical['radius'].append(np.max(np.linalg.norm(self.pop.positions - np.tile(
            current_centroid, (self.num_agents, 1)), 2, 1)))

        # TODO: Implement stagnation again
        # Update stagnation
        # if (self.pop.iteration > 0) and (
        #         float(self.historical['fitness'][-1]) == float(self.historical['fitness'][-2])):
        #     instantaneous_stagnation = self.historical['stagnation'][-1] + 1
        # else:
        #     instantaneous_stagnation = 0
        # self.historical['stagnation'].append(instantaneous_stagnation)

    def _verbose(self, text_to_print):
        """
        Print each step performed during the solution procedure. It only works if ``verbose`` flag is True.

        :param str text_to_print:
            Explanation about what the metaheuristic is doing.

        :return: None.
        """
        if self.verbose:
            print(text_to_print)
