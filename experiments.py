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
from os import path


class Experiment():
    """
    Create an experiment using the toolbox.
    """

    def __init__(self, exp_config, hh_config, prob_config):
        # Load the default experiment configuration and compare it with exp_cfg
        self.exp_config = jt.check_fields(
            {
                'experiment_name': 'test',
                'experiment_type': 'default',  # 'default' -> hh, 'brute-force', 'basic_metaheuristics'
                'heuristic_collection_file': 'default.txt',
                'weights_dataset_file': None,  # 'operators_weights.json',
                'use_parallel': True,
                'parallel_pool_size': None,  # Default
                'auto_collection_num_vals': 5
            }, exp_config)

        # Load the default hyper-heuristic configuration and compare it with hh_cfg
        self.hh_config = jt.check_fields(
            {
                'cardinality': 3,
                'num_agents': 30,
                'num_iterations': 100,
                'num_replicas': 50,
                'num_steps': 100,
                'max_temperature': 200,
                'stagnation_percentage': 0.3,
                'cooling_rate': 0.05
            }, hh_config)

        # Load the default problem configuration and compare it with prob_config
        self.prob_config = jt.check_fields(
            {
                'dimensions': [2, 5, *range(10, 50 + 1, 10)],
                'functions': bf.__all__,
                'is_constrained': True
            }, prob_config)

        # Check if the heuristic collection exists
        if not path.isfile('collections/' + exp_config['heuristic_collection_file']):
            # If the name is a reserved one. These files cannot be not created automatically
            if exp_config['heuristic_collection_file'] in ['default.txt', 'automatic.txt',
                                                           'basicmetaheuristics.txt', 'test_collection']:
                raise ExperimentError('This collection name is reserved and cannot be created automatically!')
            else:
                Operators.build_operators(Operators.obtain_operators(
                    num_vals=exp_config['auto_collection_num_vals']),
                    file_name=exp_config['heuristic_collection_file'].split('.')[0])
                self.exp_config['weights_dataset_file'] = None

        # Check if the weights dataset not exist or required
        if self.exp_config['weights_dataset_file'] and (
                self.exp_config['experiment_type'] not in ['brute_force', 'basic_metaheuristics']):
            if path.isfile('collections/' + self.exp_config['weights_dataset_file']):
                self.weights_data = jt.read_json('collections/' + self.exp_config['weights_dataset_file'])
            else:
                raise ExperimentError('A valid weights_dataset_file must be provided in exp_config')
        else:
            self.weights_data = None

    def run(self):
        # TODO: Create a task log for prevent interruptions
        # Create a list of problems from functions and dimensions combinations
        all_problems = [(x, y) for x in self.prob_config['functions'] for y in self.prob_config['dimensions']]

        # Check if the experiment will be in parallel
        if self.exp_config['use_parallel']:
            pool = multiprocessing.Pool(self.exp_config['parallel_pool_size'])
            pool.map(lambda x: self._simple_run(x), all_problems)
        else:
            for prob_dim in all_problems:
                self._simple_run(prob_dim)

    def _simple_run(self, prob_dim):
        # Read the function name and the number of dimensions
        function_string, num_dimensions = prob_dim

        # Message to print and to store in folders
        label = '{}-{}D'.format(function_string, num_dimensions)

        # Get and format the problem
        problem = eval('bf.{}({})'.format(function_string, num_dimensions))
        problem_to_solve = problem.get_formatted_problem(self.prob_config['is_constrained'])

        # Read the particular weights array (if so)
        weights = self.weights_data[str(num_dimensions)][problem.get_features(fmt='string', wrd='1')] \
            if self.weights_data else None

        # Call the hyper-heuristic object
        hh = HH.Hyperheuristic(heuristic_space=self.exp_config['heuristic_collection_file'],
                               problem=problem_to_solve, parameters=self.hh_config,
                               file_label=label, weights_array=weights)

        # Run the HH according to the specified type
        if self.exp_config['experiment_type'] == 'brute_force':
            hh.brute_force()
        elif self.exp_config['experiment_type'] == 'basic_metaheuristics':
            hh.basic_metaheuristics()
        else:
            hh.run()

        # TODO: Add pre-label to now the current status
        print(label + ' done!')


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


class ExperimentError(Exception):
    """
    Simple ExperimentError to manage exceptions.
    """
    pass


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
    pool.map(test_set1, problems_and_dimensions)
    # pool.map(test_set2p, dimensions)
