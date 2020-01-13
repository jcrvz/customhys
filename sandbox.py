# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 17:21:52 2019

@author: L03130342
"""

# Packages
import numpy as np
import benchmark_func as bf
import matplotlib.pyplot as plt
import metaheuristic

#%% Problem definition
num_dimensions = 2
num_agents = 50
num_iterations = 100

problem = bf.Sphere(num_dimensions)
#problem = bf.Rosenbrock(num_dimensions)
#problem = bf.Ackley(num_dimensions)
#problem = bf.Griewank(num_dimensions)

is_constrained = True

# Find the problem function : objective function to minimise
problem_function = lambda x : problem.get_func_val(x)

# Define the problem domain
boundaries = (0.01*problem.min_search_range, 0.01*problem.max_search_range)

#%% Build the simple heuristic collection (if so)

search_operators = [('swarm_dynamic', {'factor': 1.1, 'self_conf': 2.54, 'swarm_conf': 2.56, 'version': 'constriction'}, 'all')]
mh = metaheuristic.Metaheuristic({
    'function': problem_function, 'boundaries': boundaries,
    'is_constrained': True}, search_operators, num_agents, num_iterations)
