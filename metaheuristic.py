# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 16:56:01 2019

@author: Jorge Mario Cruz-Duarte (jcrvz.github.io)
"""

import numpy as np
from population import Population as pop
import operators as op
import matplotlib.pyplot as plt


# Read all available operators
__operators__ = op.__all__  # [x[0] for x in op._obtain_operators(1)]
__selectors__ = ['greedy', 'probabilistic', 'metropolis', 'all', 'none']


class Metaheuristic():
    def __init__(self, problem, search_operators, num_agents=30):
        """
        Create a metaheuristic method by employing different simple search
        operators.

        Parameters
        ----------
        problem : dict
            This is a dictionary containing the 'function' that maps a 1-by-D
            array of real values ​​to a real value, 'is_constrained' flag
            indicated that solution is inside the search space, and the
            'boundaries' (a tuple with two lists of size D). These two lists
            correspond to the lower and upper limits of search space, such as:
                boundaries = (lower_boundaries, upper_boundaries)
            Note: Dimensions of search domain are read from these boundaries.
        search_operators : list
            A list of available search operators.
        num_agents : int, optional
            Numbre of agents or population size. The default is 30.

        Returns
        -------
        None.

        """
        # Define the problem function
        self.problem_function = problem['function']

        # Create population
        self.pop = pop.Population(problem['boundaries'],
                                  num_agents, problem['is_constrained'])

        # Check and read the search_operators
        self.operators, self.selectors = op._process_operators(
            search_operators)

        # Define the maximum number of iterations
        self.num_iterations = 100

        # Read the number of dimensions
        self.num_dimensions = self.pop.num_dimensions

        # Read the number of agents
        self.num_agents = num_agents

        # Initialise historical variables
        self.historical = dict()

        # Set additional variables
        self.verbose = True

    def run(self):
        """
        Run the metaheuristic for solving a defined problem.

        Returns
        -------
        None.

        """
        # Set initial iteration
        self.pop.iteration = 0

        # Initialise the population
        self.pop.initialise_positions()  # Default: random

        # Evaluate fitness values
        self.pop.evaluate_fitness(self.problem_function)

        # Update population, particular, and global
        self.pop.update_positions()  # Default: 'population', 'all'
        self.pop.update_positions('particular', 'all')
        self.pop.update_positions('global', 'greedy')  # Default: greedy

        # Initialise and update historical variables
        self._reset_historicals()
        self._update_historicals()

        # Start optimisaton procedure
        for iteration in range(1, self.num_iterations + 1):
            # Update the current iteration
            self.pop.iteration = iteration

            self._verbose("\nIteration {}:\n{}".format(iteration, '-' * 50))

            # Implement the sequence of operators and selectors
            for operator, selector in zip(self.operators, self.selectors):
                # Apply an operator
                exec("op." + operator)

                # Evaluate fitness values
                self.pop.evaluate_fitness(self.problem_function)

                # Update population
                if selector in __selectors__:
                    self.pop.update_positions('population', selector)
                else:
                    self.pop.update_positions()

                # Update global position
                self.pop.update_positions('global', 'greedy')

                # Report change
                self._verbose("{} and {} selection applied!".format(
                    operator, selector))

            # Update historical variables
            self._update_historicals()

            # Verbose (if so) some information
            self._verbose("Stag. counter: {}, pop. radious: {}".format(
                self.historical_stagnation[-1], self.historical_radius[-1]))
            self._verbose(self.pop.get_state())

    def get_solution(self):
        """
        Deliver the last position and fitness obtained after run.

        Returns
        -------
        ndarray
            Best position vector found.
        float
            Best fitness value found.
        """
        return self.historical['position'][-1], self.historical['fitness'][-1]

    def show_performance(self):
        """
        Show the solution evolution during the iterative process.

        Returns
        -------
        None.

        """
        # Show historical fitness
        fig1, ax1 = plt.subplots()

        color = 'tab:red'
        plt.xlabel("Iterations")
        ax1.set_ylabel("Global Fitness", color=color)
        ax1.plot(np.arange(0, self.num_iterations + 1),
                 self.historical['fitness'], color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_yscale('linear')

        ax2 = ax1.twinx()

        color = 'tab:blue'
        ax2.set_ylabel('Population radius', color=color)
        ax2.plot(np.arange(0, self.num_iterations + 1),
                 self.historical['radius'], color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_yscale('log')

        fig1.tight_layout()
        plt.show()

    def _reset_historicals(self):
        """
        Reset the self.historical variable

        Returns
        -------
        None.

        """
        self.historical = dict(
            fitness=list(),
            position=list(),
            centroid=list(),
            radius=list(),
            stagnation=list(),
            )

    def _update_historicals(self):
        """
        Update the historical variables

        Returns
        -------
        None.

        """
        # Update historical variables
        self.historical['fitness'].append(self.pop.global_best_fitness)
        self.historical['position'].append(self.pop.global_best_position)

        # Update population centroid and radius
        current_centroid = np.array(self.pop.positions).mean(0)
        self.historical['centroid'].append(current_centroid)
        self.historical['radius'].append(np.linalg.norm(
            self.pop.positions - np.tile(current_centroid,
                                         (self.num_agents, 1)), 2, 1).max())

        # Update stagnation
        if (self.pop.iteration > 0) and (
                float(self.historical['fitness'][-2:]) == 0.0):
            instantaneous_stagnation = self.historical['stagnation'][-1] + 1
        else:
            instantaneous_stagnation = 0
        self.historical['stagnation'].append(instantaneous_stagnation)

    def _verbose(self, text_to_print):
        """
        Print each step performed during the solution procedure

        Parameters
        ----------
        text_to_print : str
            Explanation about what the metaheuristic is doing.

        Returns
        -------
        None.

        """
        if self.verbose:
            print(text_to_print)
