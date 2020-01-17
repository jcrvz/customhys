# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 13:42:15 2019

@author: L03130342
"""

import hyperheuristic as HH
from metaheuristic import Operators
#from metaheuristic import Population
import numpy as np
import benchmark_func as bf
# import matplotlib.pyplot as plt

# %% Problems definition

dimensions = [2, 10, 30]
functions = ['Rosenbrock', 'Griewank']  # 'Sphere', 'Ackley', 
divider = 1.0
is_constrained = True

# %% Hyperheuristic conditions
hh_parameters = {
    'cardinality': 2,
    'num_agents': 30,
    'num_iterations': 100,
    'num_replicas': 100,
    'num_steps': 5,
    'num_trials': 50
    }

# %% Generate the search operator collection (once)
# Operators._build_operators()  # <- uncomment

heuristics_collection = 'operators_collection.txt'  # full
# heuristics_collection = 'single_test.txt'  # just 1

# %% Find a metaheuristic for each problem

for num_dimensions in dimensions:
    for function_string in functions:
        label = f"{function_string}-{num_dimensions}D"

        problem = eval(f"bf.{function_string}({num_dimensions})")
        function = lambda x: problem.get_func_val(x)

        # HH.set_problem(problem_function, boundaries, True)
        Problem = HH.set_problem(
            function,
            (problem.min_search_range/divider,
             problem.max_search_range/divider),
            is_constrained
            )

        hh = HH.Hyperheuristic(heuristics_collection, Problem, hh_parameters,
                               label)

        hh.run()
