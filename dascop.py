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
import multiprocessing

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
            Problem = HH.set_problem(function,
                (problem.min_search_range/divider, problem.max_search_range/divider),
                is_constrained)

            # Call the hyperheuristic object
            hh = HH.Hyperheuristic(heuristics_collection, Problem, hh_parameters, label)

            # Run the HH:Random Search
            hh.run()

# %% Test set used for evaluate all the search operators in the collection
def test_set1():
    # Problems definition
    dimensions = [2, *range(5, 50+1, 5)]
    functions = bf.__all__
    divider = 1.0
    is_constrained = True

    # Hyperheuristic conditions
    hh_parameters = {
        'cardinality': 1,
        'num_agents': 30,
        'num_iterations': 100,
        'num_replicas': 10,
        'num_trials': 100,       # Not used
        'max_temperature': 100,  # Not used
        'min_temperature': 0.1,  # Not used
        'cooling_rate': 0.05,    # Not used
        }

    # Generate the search operator collection (once)
    Operators._build_operators(
        Operators._obtain_operators(num_vals=3), file_name="automatic")

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


# %% Parallel of test_set1()
def test_set1p(num_dimensions):
    # Problems definition
    functions = bf.__all__  # [82] [20]
    divider = 1.0
    is_constrained = True

    # Hyperheuristic conditions
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

    # print('-' * 10)
    # Find a metaheuristic for each problem
    # for num_dimensions in dimensions:
        # print('Dim: {}/{},'.format(
        #     num_dimensions-1, len(dimensions)), end=' ')
    if isinstance(functions, str):
        functions = [functions]
    for func_id in range(len(functions)):
        function_string = functions[func_id]

        # print('Func: {}/{}...'.format(func_id + 1, len(functions)))

        # Message to print and to store in folders
        label = "{}-{}D".format(function_string, num_dimensions)
        # print('... ' + label + ':')

        # Format the problem
        problem = eval("bf.{}({})".format(function_string, num_dimensions))

        # HH.set_problem(problem_function, boundaries, True)
        problem_to_solve = HH.set_problem(
            lambda x: problem.get_function_value(x),
            (problem.min_search_range/divider,
             problem.max_search_range/divider),
            is_constrained
            )

        # Call the hyperheuristic object
        hh = HH.Hyperheuristic('default.txt', problem_to_solve, hh_parameters, label)

        # Run the HH:Random Search
        hh.brute_force()

        print(label + " done!")


# %% Parallel try of test_set2()
def test_set2p(num_dimensions):
    # Problems definition
    functions = bf.__all__  # all
    divider = 1.0
    is_constrained = True

    # Hyperheuristic conditions
    hh_parameters = {
        'cardinality': 2,
        'num_agents': 30,
        'num_iterations': 100,
        'num_replicas': 30,
        'num_steps': 10,
        'max_temperature': 100,
        'min_temperature': 0.1,
        'cooling_rate': 0.5,
        }

    # print('-' * 10)
    # Find a metaheuristic for each problem
    # for num_dimensions in dimensions:
        # print('Dim: {}/{},'.format(
        #     num_dimensions-1, len(dimensions)), end=' ')
    if isinstance(functions, str):
        functions = [functions]

    for func_id in range(len(functions)):
        function_string = functions[func_id]

        # print('Func: {}/{}...'.format(func_id + 1, len(functions)))

        # Message to print and to store in folders
        label = "{}-{}D".format(function_string, num_dimensions)
        # print('... ' + label + ':')

        # Format the problem
        problem = eval("bf.{}({})".format(function_string, num_dimensions))

        # HH.set_problem(problem_function, boundaries, True)
        problem_to_solve = HH.set_problem(lambda x: problem.get_function_value(x),
            (problem.min_search_range/divider, problem.max_search_range/divider),
            is_constrained)

        # Call the hyperheuristic object
        hh = HH.Hyperheuristic('test-set-21.txt', problem_to_solve, hh_parameters, label)

        # Run the HH:Random Search
        hh.run()

        print(label + " done!")

# %% Autorun
if __name__ == '__main__':
    # Build the collection of operators
    Operators._build_operators(Operators._obtain_operators(num_vals=21), file_name="test-set-21")

    dimensions = [2, 5, *range(10, 50 + 1, 10)]

    # Run it in parallel
    pool = multiprocessing.Pool()
    pool.map(test_set2p, dimensions)
    # pool.join()
    # pool.close()
