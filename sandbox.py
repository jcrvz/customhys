# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 17:21:52 2019

@author: L03130342
"""

# Packages
import numpy as np
import dascop as dso
import benchmark_func as bf
import matplotlib.pyplot as plt


#%% Problem definition
num_dimensions = 2
num_agents = 50
num_iterations = 100

#problem = bf.Sphere(num_dimensions)
problem = bf.Rosenbrock(num_dimensions)
#problem = bf.Ackley(num_dimensions)
#problem = bf.Griewank(num_dimensions)

is_constrained = True

# Find the problem function : objective function to minimise
problem_function = lambda x : problem.get_func_val(x)

# Define the problem domain
boundaries = (problem.min_search_range, problem.max_search_range)

#%% Build the simple heuristic collection (if so)
dso.heuristic_configurations()

# For a given set of simple heuristics return a metaheuristic