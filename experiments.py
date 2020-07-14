# -*- coding: utf-8 -*-
"""
This script contains the main experiments performed using the current framework.

Created on Mon Sep 30 13:42:15 2019

@author: Jorge Mario Cruz-Duarte (jcrvz.github.io), e-mail: jorge.cruz@tec.mx
"""

import hyperheuristic as HH
from metaheuristic import Operators
import benchmark_func as bf
import multiprocessing
import tools as jt


# %% Parallel of test_set1() : Brute-force
def test_set0(num_dimensions):
    # Problems definition
    functions = bf.__all__
    is_constrained = True

    # Hyperheuristic conditions
    hh_parameters = {
        'cardinality': 1,
        'num_agents': 30,
        'num_iterations': 100,
        'num_replicas': 30,
        'num_trials': 100,  # Not used
        'max_temperature': 100,  # Not used
        'min_temperature': 0.1,  # Not used
        'cooling_rate': 0.05,  # Not used
    }

    # Find a metaheuristic for each problem
    if isinstance(functions, str):  # If it is only one function
        functions = [functions]
    for func_id in range(len(functions)):
        function_string = functions[func_id]

        # Message to print and to store in folders
        label = "{}-{}D".format(function_string, num_dimensions)

        # Get and format the problem
        problem = eval("bf.{}({})".format(function_string, num_dimensions))
        problem_to_solve = problem.get_formatted_problem(is_constrained)

        # Call the hyperheuristic object
        hh = HH.Hyperheuristic('default.txt', problem_to_solve, hh_parameters, label)

        # Run the HH:Random Search
        hh.brute_force()

        print(label + " done!")

# %% Parallel of test_set1() : Basic Heuristics
def test_set1(problem_dimension):

    function_string, num_dimensions = problem_dimension
    is_constrained = True

    # Hyperheuristic conditions
    hh_parameters = {
        'cardinality': 1,  # Does not matter
        'num_agents': 30,
        'num_iterations': 100,
        'num_replicas': 30,
        'num_trials': 100,  # Not used
        'max_temperature': 100,  # Not used
        'min_temperature': 0.1,  # Not used
        'cooling_rate': 0.05,  # Not used
    }

    # if isinstance(functions, str):
    #     functions = [functions]
    # for func_id in range(len(functions)):
    #     function_string = functions[func_id]

    # print('Func: {}/{}...'.format(func_id + 1, len(functions)))

    # Message to print and to store in folders
    label = '{}-{}D'.format(function_string, num_dimensions)

    # Get and format the problem
    problem = eval('bf.{}({})'.format(function_string, num_dimensions))
    problem_to_solve = problem.get_formatted_problem(is_constrained)

    # Call the hyperheuristic object
    hh = HH.Hyperheuristic('basicmetaheuristics.txt', problem_to_solve, hh_parameters, label)

    # Run the HH:Random Search
    hh.basic_metaheuristics(label)

    print(label + ' done!')


# %% Parallel try of test_set2() After brute-force
def test_set2(problem_dimension):

    function_string, num_dimensions = problem_dimension

    # Problems definition
    # functions = bf.__all__
    weights_per_feature = weights_data[str(num_dimensions)]
    is_constrained = True

    # Hyperheuristic conditions
    hh_parameters = {
        'cardinality': 3,
        'num_agents': 30,
        'num_iterations': 100,
        'num_replicas': 50,
        'num_steps': 100,
        'max_temperature': 200,
        'stagnation_percentage': 0.3,
        'cooling_rate': 0.05,
    }

    # Message to print and to store in folders
    label = '{}-{}D'.format(function_string, num_dimensions)

    # Get and format the problem
    problem = eval('bf.{}({})'.format(function_string, num_dimensions))
    problem_to_solve = problem.get_formatted_problem(is_constrained)

    # Call the hyperheuristic object
    hh = HH.Hyperheuristic('test-set-11.txt', problem_to_solve, hh_parameters, label, None)
                           # weights_per_feature[problem.get_features(fmt='string', wrd='1')])

    # Run the HH:Random Search
    hh.run()

    print(label + ' done!')

# %% Parallel try of test_set2() After brute-force
def test_set3(problem_dimension):

    function_string, num_dimensions = problem_dimension

    # Problems definition
    # functions = bf.__all__
    # weights_per_feature = weights_data[str(num_dimensions)]
    is_constrained = True

    # Hyperheuristic conditions
    hh_parameters = {
        'cardinality': 3,
        'num_agents': 30,
        'num_iterations': 100,
        'num_replicas': 50,
        'num_steps': 100,
        'max_temperature': 200,
        'stagnation_percentage': 0.3,
        'cooling_rate': 0.05,
    }

    # Message to print and to store in folders
    label = '{}-{}D'.format(function_string, num_dimensions)

    # Get and format the problem
    problem = eval('bf.{}({})'.format(function_string, num_dimensions))
    problem_to_solve = problem.get_formatted_problem(is_constrained)

    # Call the hyperheuristic object
    hh = HH.Hyperheuristic('test-set-11.txt', problem_to_solve, hh_parameters, label, None)
                           # weights_per_feature[problem.get_features(fmt='string', wrd='1')])

    # Run the HH:Random Search
    hh.run()

    print(label + ' done!')



# %% Auto-run
if __name__ == '__main__':
    # List of dimensionalities
    dimensions = [2, 5, *range(10, 50 + 1, 10)]

    # Build the collection of operators
    # Operators.build_operators(Operators.obtain_operators(num_vals=11), file_name="test-set-11")
    functions = bf.__all__

    problems_and_dimensions = [(x, y) for x in functions for y in dimensions]

    # Load the weight data
    # weights_data = jt.read_json('collections/operators_weights.json')

    # Run it in parallel
    pool = multiprocessing.Pool(10)
    pool.map(test_set3p, problems_and_dimensions)
    # pool.map(test_set2p, dimensions)
