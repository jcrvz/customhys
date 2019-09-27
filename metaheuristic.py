# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 16:56:01 2019

@author: L03130342
"""

import numpy as np
from population import Population, __selection__

# %% --------------------------------------------------------------------------
class Metaheuristic():
    # Internal variables
    
    # Class initialisation
    # -------------------------------------------------------------------------
    def __init__(self, problem_function, boundaries, simple_heuristics, 
                 is_constrained = True, num_agents = 30, 
                 threshold_iterations = 100):
#                 threshold_stagnation = 30, threshold_fitness_change = 1e-12, 
#                 threshold_position_change = 1e-12, 
#                 threshold_fitness = 1e-12, theoretical_fitness = 0.0
#                 threshold_population_radius = 1e-3):             
        
        # Create population
        self.pop = Population(problem_function, boundaries, num_agents, 
                              is_constrained)        
        
        # Check and read the simple heuristics
        self.operators, self.selectors = self.__process_heuristics(
                simple_heuristics)
        
        # Define the maximum number of iterations
        self.num_iterations = threshold_iterations
        
        # Initialise historical variables
        
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
        self.pop.update_global()             # Default: greedy
        
        # Start optimisaton procedure
        for iteration in range(1, self.num_iterations + 1):
            # Update the current iteration
            self.pop.iteration = iteration
            
            # Implement the sequence of operators and selectors
            for operator, selector in zip(self.operators,self.selectors):
                # Apply an operator
                exec("self.pop." + operator)
                
                # Evaluate fitness values
                self.pop.evaluate_fitness() 
                
                # Update population 
                if selector in __selection__:
                    self.pop.update_population(selector)
                else:
                    self.pop.update_population("all")
                
                # Update global position
                self.pop.update_global() 
                
                # Report change
                print(f"operator {operator} with selector {selector} were applied!")
            
            print("[",iteration,"] -> ",self.pop.get_state())
        
    # Process simple heuristics entered as list of tuples
    # -------------------------------------------------------------------------
    def __process_heuristics(self, simple_heuristics):
        # Initialise the list of callable operators (simple heuristics)
        executable_operators = []; selectors = []
        
        # For each simple heuristic, read their parameters and values
        for operator, parameters, selector in simple_heuristics:
            # Store selectors
            selectors.append(selector)
            
            if len(parameters) >= 0:
                sep = ","; str_parameters = []
                
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