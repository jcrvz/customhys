# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 15:36:43 2020

@author: Jorge Mario Cruz-Duarte (jcrvz.github.io)
"""
import numpy as np
import scipy.stats as st
# import scipy as sp
from metaheuristic import Metaheuristic
# from metaheuristic import Population
# from metaheuristic import Operators
from datetime import datetime
import json
from os.path import exists as _check_path
from os import makedirs as _create_path
from tqdm import tqdm

# sp.np.seterr(divide='ignore', invalid='ignore')
# np.seterr(divide='ignore', invalid='ignore')

class Hyperheuristic():
    def __init__(self, heuristic_space, problem, parameters={
            'cardinality': 2, 'num_iterations': 100, 'num_agents': 30,
            'num_replicas': 100, 'num_steps': 100}):
        # Read the heuristic space
        if isinstance(heuristic_space, list):
            self.heuristic_space = heuristic_space
        elif isinstance(heuristic_space, str):
            with open(heuristic_space, 'r') as operators_file:
                self.heuristic_space = [
                    eval(line.rstrip('\n')) for line in operators_file]
        else:
            HyperheuristicError("Invalid heuristic_space")

        # Read the heuristic space (mandatory)
        self.problem = problem
        self.num_operators = len(self.heuristic_space)

        # Initialise other parameters
        self.parameters = parameters

    def run(self):
        """
        Run Random Search (integer version) to find the best metaheuristic.
        Each meatheuristic is run 'num_replicas' times to obtain statistics
        and thus its performance.

        Returns
        -------
        solution : list
            The sequence of search operators that compose the metaheuristic.
        performance : float
            This is a metric defined in get_performance
        encoded_solution : list
            The sequence of indices that correspond to the search operators.
        historicals : dict
            A dictionary of information from each step. Its keys are:
            'step', 'encoded_solution', 'solution', 'performances', and
            'details'.
            This later field, 'details', is also a dictionary which contains
            information about each replica carried out with the metaheuristic.
            Its fields are 'historical' (each iteration that the metaheuristic
            has performed), 'fitness', 'positions', and 'statistics'.

        """
        # Create the initial solution
        encoded_solution = np.random.randint(
            0, self.num_operators, self.parameters['cardinality'])
        solution = [self.heuristic_space[index] for index in encoded_solution]

        # Evaluate this solution
        performance, details = self.evaluate_metaheuristic(solution)

        # Initialise historical register
        historicals = dict(
            encoded_solution=[encoded_solution],
            solution=[solution],
            performances=[performance],
            details=[details])

        # Save this historical register
        _save_iteration(0, historicals)
        
        print('{} - perf: {}, sol: {}'.format(
            0, performance, encoded_solution))

        # Perform the random search
        for step in range(1, self.parameters['num_steps'] + 1):
            # tqdm(range(1, self.parameters['num_steps'] + 1),
            #                   desc='HH', position = 0, leave = True,
            #                   postfix = {'performance': performance},
            #                   bar_format="{l_bar}{bar}| " +
            #                   "[{n_fmt}/{total_fmt}" + "{postfix}]"):

            # Select randomly a candidate solution
            encoded_candidate_solution = np.random.randint(
                0, self.num_operators, self.parameters['cardinality'])
            candidate_solution = [self.heuristic_space[index] for index in
                                  encoded_candidate_solution]

            # Evaluate this candidate solution
            candidate_performance, candidate_details =\
                self.evaluate_metaheuristic(candidate_solution)

            # Check improvement (greedy selection)
            if candidate_performance < performance:
                encoded_solution = encoded_candidate_solution
                solution = candidate_solution
                performance = candidate_performance
                details = candidate_details
                
                print('{} - perf: {}, sol: {}'.format(
                    step, performance, encoded_solution))

                # Update historicals (only if improves)
                # historicals['step'].append(step)
                # historicals['encoded_solution'].append(encoded_solution)
                # historicals['solution'].append(solution)
                # historicals['performance'].append(performance)
                # historicals['details'].append(candidate_details)

                # Save this historical register
                _save_iteration(step, {
                    'encoded_solution': encoded_solution,
                    'solution': solution,
                    'performance': performance,
                    'details': details})

        return solution, performance, encoded_solution, historicals

    def evaluate_metaheuristic(self, search_operators):
        # Initialise the historical registers
        historical_data = list()
        fitness_data = list()
        position_data = list()

        # Run the metaheuristic several times
        for rep in tqdm(range(1, self.parameters['num_replicas'] + 1),
                        desc='--MH',
                        position = 0, leave = True,
                        postfix = {'fitness':
                                   fitness_data[-1] if
                                   len(fitness_data) != 0 else '?'},
                            bar_format="{l_bar}{bar}| " +
                            "[{n_fmt}/{total_fmt}" + "{postfix}]"):
            
            # Call the metaheuristic
            mh = Metaheuristic(self.problem, search_operators,
                               self.parameters['num_agents'],
                               self.parameters['num_steps'])

            # Run this metaheuristic
            mh.run()

            # Store the historical values from this run
            historical_data.append(mh.historical)

            # Read and store the solution obtained
            _temporal_position, _temporal_fitness = mh.get_solution()
            fitness_data.append(_temporal_fitness)
            position_data.append(_temporal_position)

        # Determine a performance metric once finish the repetitions
        fitness_stats = self.get_statistics(fitness_data)

        return self.get_performance(fitness_stats), dict(
            historical=historical_data, fitness=fitness_data,
            positions=position_data, statistics=fitness_stats)

    @staticmethod
    def get_performance(statistics):
        # Score function using the statistics from fitness values
        perf = statistics['Med']  # Option 1
        # perf = statistics['Avg'] + statistics['Std']  # Option 2
        # perf = statistics['Med'] + statistics['IQR']  # Option 3
        # perf = statistics['Avg'] + statistics['Std'] + \
        # statistics['Med'] + statistics['IQR']
        return perf

    @staticmethod
    def get_statistics(raw_data):
        # Get descriptive statistics
        dst = st.describe(raw_data)

        # Store statistics
        return dict(nob=dst.nobs,
                    Min=dst.minmax[0],
                    Max=dst.minmax[1],
                    Avg=dst.mean,
                    Std=np.std(raw_data),
                    Skw=dst.skewness,
                    Kur=dst.kurtosis,
                    IQR=st.iqr(raw_data),
                    Med=np.median(raw_data),
                    MAD=st.median_absolute_deviation(raw_data))


def _save_iteration(iteration_number, variable_to_save):
    # Get the current date
    now = datetime.now()

    # Define the folder name
    folder_name = "raw_data/" + now.strftime("%m_%d_%Y")

    # Check if this path exists
    if not _check_path(folder_name):
        _create_path(folder_name)

    # Create a new file for this step
    with open(folder_name + f"/{iteration_number}-" + now.strftime(
            "%H_%M_%S") + ".json", 'w') as json_file:
        json.dump(variable_to_save, json_file, cls=NumpyEncoder)


def set_problem(function, boundaries, is_constrained=True):
    return {'function': function, 'boundaries': boundaries,
            'is_constrained': is_constrained}


class HyperheuristicError(Exception):
    """
    Simple HyperheuristicError to manage exceptions.
    """
    pass

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)