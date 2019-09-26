# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 16:56:01 2019

@author: L03130342
"""

import numpy as np
from population import Population

# %% --------------------------------------------------------------------------
class Metaheuristic():
    # Internal variables
    
    # Class initialisation
    def __init__(self, problem_function, boundaries, simple_heuristics, 
                 is_constrained = True, num_agents = 30, 
                 desired_performance = 1e-6):             
        
        # Create population
        self.pop = Population(problem_function, boundaries, num_agents, 
                              is_constrained)        
        
        # Check and read the simple heuristics
        self.operators = self.__process_heuristics(simple_heuristics)
    
    def __process_heuristics(self, simple_heuristics):
        # Initialise the list of callable operators (simple heuristics)
        executable_operators = []
        
        # For each simple heuristic, read their parameters and values
        for operator, parameters in simple_heuristics:
            if len(parameters) >= 0:
                sep = ","; str_parameters = []
                
                for parameter, value in parameters.items():
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
        
        return executable_operators