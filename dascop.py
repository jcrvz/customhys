# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 13:42:15 2019

@author: L03130342
"""

import hyperheuristic as HH
#import numpy as np
import benchmark_func as bf
#import matplotlib.pyplot as plt


# %% Problem definition
num_dimensions = 2

#problem = bf.Sphere(num_dimensions)
problem = bf.Rosenbrock(num_dimensions)
#problem = bf.Ackley(num_dimensions)
#problem = bf.Griewank(num_dimensions)

is_constrained = True

# Find the problem function : objective function to minimise
problem_function = lambda x : problem.get_func_val(x)

# Define the problem domain
boundaries = (problem.min_search_range, problem.max_search_range)

# %% Initialise the hyperheuristic procedure

# heuristics_collection = 'test_collection.txt'
heuristics_collection = 'single_test.txt'

problem_info = HH.set_problem(problem_function, boundaries, True)

hh = HH.Hyperheuristic(heuristics_collection, problem_info, {
    'cardinality': 2,
    'num_agents': 30,
    'num_iterations': 10,
    'num_replicas': 2,
    'num_steps': 10})

hh.run()
