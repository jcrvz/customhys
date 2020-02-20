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

# %% Test set used for A Primary Study on Hyper-Heuristics to Customise
#      Metaheuristics for Continuous Optimisation, submitted to CEC'20.
def test_set0():
    # Problems definition
    dimensions = [2 , 10, 30]
    functions = ['Griewank','Ackley', 'Rosenbrock', 'Sphere']
    divider = 1.0
    is_constrained = True

    # Hyperheuristic conditions (it only works with 1st dascop)
    hh_parameters = {
        'cardinality': 3,
        'num_agents': 30,
        'num_iterations': 100,
        'num_replicas': 100,
        'num_steps': 5,
        'num_trials': 50
        }

    # Generate the search operator collection (once)
    # Operators._build_operators()  # <- uncomment

    heuristics_collection = 'operators_collection.txt'  # full
    # heuristics_collection = 'single_test.txt'  # just 1

    # Find a metaheuristic for each problem
    for num_dimensions in dimensions:
        for function_string in functions:
            # Message to print
            label = f"{function_string}-{num_dimensions}D"
            print('... ' + label + ':')

            # Format the problem
            problem = eval(f"bf.{function_string}({num_dimensions})")
            function = lambda x: problem.get_func_val(x)
            # HH.set_problem(problem_function, boundaries, True)
            Problem = HH.set_problem(
                function,
                (problem.min_search_range/divider,
                 problem.max_search_range/divider),
                is_constrained
                )

            # Call the hyperheuristic object
            hh = HH.Hyperheuristic(heuristics_collection, Problem,
                                   hh_parameters, label)

            # Run the HH:Random Search
            hh.run()

# %% Test set used for evaluate all the search operators in the collection
def test_set1():
    # Problems definition
    dimensions = range(2, 30 + 1)
    functions = bf.__all__
    divider = 1.0
    is_constrained = True

    # Hyperheuristic conditions (it only works with 1st dascop)
    hh_parameters = {
        'cardinality': 1,
        'num_agents': 30,
        'num_iterations': 100,
        'num_replicas': 30,
        'num_trials': 100,       # Not used
        'max_temperature': 100,  # Not used
        'min_temperature': 0.1,  # Not used
        'cooling_rate': 0.05,    # Not used
        }

    # Generate the search operator collection (once)
    Operators._build_operators(
        Operators._obtain_operators(num_vals=5), file_name="automatic")

    heuristics_collection = 'automatic.txt'

    print('-' * 10)
    # Find a metaheuristic for each problem
    for num_dimensions in dimensions:
        print('Dim: {}/{},'.format(
            num_dimensions-1, len(dimensions)), end=' ')
        for func_id in range(len(functions)):
            function_string = functions[func_id]

            print('Func: {}/{}...'.format(func_id + 1, len(functions)))

            # Message to print and to store in folders
            label = "{}-{}D".format(function_string, num_dimensions)
            print('... ' + label + ':')

            # Format the problem
            problem = eval("bf.{}({})".format(function_string, num_dimensions))

            # HH.set_problem(problem_function, boundaries, True)
            Problem = HH.set_problem(
                lambda x: problem.get_function_value(x),
                (problem.min_search_range/divider,
                 problem.max_search_range/divider),
                is_constrained
                )

            # Call the hyperheuristic object
            hh = HH.Hyperheuristic(heuristics_collection, Problem,
                                   hh_parameters, label)

            # Run the HH:Random Search
            hh.brute_force()

# %% Autorun
if __name__ == '__main__':
    test_set1()