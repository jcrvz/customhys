# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 16:56:01 2019

@author: L03130342
"""

import numpy as np
import population as pop
import matplotlib.pyplot as plt

class Metaheuristic():
    # Internal variables

    # Class initialisation
    # -------------------------------------------------------------------------
    def __init__(self, problem_function, boundaries, simple_heuristics, is_constrained=True, num_agents=30,
                 threshold_iterations=100, verbose=True):
        #                 threshold_stagnation = 30, threshold_fitness_change = 1e-12,
        #                 threshold_position_change = 1e-12,
        #                 threshold_fitness = 1e-12, theoretical_fitness = 0.0
        #                 threshold_population_radius = 1e-3):

        # Create population
        self.pop = pop.Population(problem_function, boundaries, num_agents,
                                  is_constrained)

        # Check and read the simple heuristics
        self.operators, self.selectors = self.process_heuristics(simple_heuristics)

        # Define the maximum number of iterations
        self.num_iterations = threshold_iterations

        # Read the number of dimensions
        self.num_dimensions = self.pop.num_dimensions

        # Read the number of agents
        self.num_agents = num_agents

        # Initialise historical variables
        self.historical_global_fitness = list()
        self.historical_global_position = list()
        self.historical_centroid = list()
        self.historical_radius = list()
        self.historical_stagnation = list()

        # Set additional variables
        self.verbose = verbose

    def process_heuristics(self, simple_heuristics):
        # Initialise the list of callable operators (simple heuristics)
        executable_operators = []
        selectors = []

        # For each simple heuristic, read their parameters and values
        for operator, parameters, selector in simple_heuristics:
            # Store selectors
            selectors.append(selector)

            if len(parameters) >= 0:
                sep = ","
                str_parameters = []

                for parameter, value in parameters.items():

                    # Check if a value is string
                    if type(value) == str:
                        str_parameters.append(f"{parameter} = '{value}'")
                    else:
                        str_parameters.append(f"{parameter} = {value}")

                # Create an executable string with given arguments
                full_string = f"{operator}({sep.join(str_parameters)})"
            else:
                # Create an executable string with default arguments
                full_string = f"{operator}()"

            # Store the read operator
            executable_operators.append(full_string)

        return executable_operators, selectors

    # Run the metaheuristic search
    # -------------------------------------------------------------------------
    def run(self):
        # Set initial iteration
        self.pop.iteration = 0

        # Initialise the population
        self.pop.initialise_uniformly()

        # Evaluate fitness values
        self.pop.evaluate_fitness()

        # Update population, particular, and global
        self.pop.update_population("all")
        self.pop.update_particular("all")
        self.pop.update_global()  # Default: greedy

        # Update historical variables
        self.__update_historicals()

        # Start optimisaton procedure
        for iteration in range(1, self.num_iterations + 1):
            # Update the current iteration
            self.pop.iteration = iteration

            self.__verbose(f"\nIteration {iteration}:\n{'-' * 50}")

            # Implement the sequence of operators and selectors
            for operator, selector in zip(self.operators, self.selectors):
                # Apply an operator
                exec("self.pop." + operator)

                # Evaluate fitness values
                self.pop.evaluate_fitness()

                # Update population 
                if selector in pop.__selection__:
                    self.pop.update_population(selector)
                else:
                    self.pop.update_population("all")

                # Update global position
                self.pop.update_global()

                # Report change
                self.__verbose(f"{operator} and {selector} selection were "
                               + f"applied!")

            # Update historical variables
            self.__update_historicals()

            # Verbose (if so) some information
            self.__verbose("Stagnation counter: " +
                           f"{self.historical_stagnation[-1]}, population radious: " +
                           f"{self.historical_radius[-1]}")
            self.__verbose(self.pop.get_state())

    # Show historical variables
    # -------------------------------------------------------------------------   
    def show_performance(self):
        # Show historical fitness
        fig1, ax1 = plt.subplots()

        color = 'tab:red'
        plt.xlabel("Iterations")
        ax1.set_ylabel("Global Fitness", color=color)
        ax1.plot(np.arange(0, self.num_iterations + 1),
                 self.historical_global_fitness, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_yscale('linear')

        ax2 = ax1.twinx()

        color = 'tab:blue'
        ax2.set_ylabel('Population radius', color=color)
        ax2.plot(np.arange(0, self.num_iterations + 1), self.historical_radius,
                 color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_yscale('log')

        fig1.tight_layout()
        plt.show()

    # Update historical variables
    # -------------------------------------------------------------------------
    def __update_historicals(self):
        # Update historical variables
        self.historical_global_fitness.append(self.pop.global_best_fitness)
        self.historical_global_position.append(self.pop.global_best_position)

        # Update population centroid and radius
        self.historical_centroid = np.array(self.pop.positions).mean(0)
        self.historical_radius.append(np.linalg.norm(self.pop.positions -
                                                     np.tile(self.historical_centroid, (self.num_agents, 1)), 2,
                                                     1).max())

        # Update stagnation
        if (self.pop.iteration > 0) and (self.historical_global_fitness[-2] ==
                                         self.historical_global_fitness[-1]):
            instantaneous_stagnation = self.historical_stagnation[-1] + 1
        else:
            instantaneous_stagnation = 0
        self.historical_stagnation.append(instantaneous_stagnation)

    # Print if verbose flag is active
    # -------------------------------------------------------------------------
    def __verbose(self, text_to_print):
        if self.verbose:
            print(text_to_print)

    # Process simple heuristics entered as list of tuples
    # -------------------------------------------------------------------------
