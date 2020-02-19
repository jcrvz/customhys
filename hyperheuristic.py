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
# from tqdm import tqdm

# sp.np.seterr(divide='ignore', invalid='ignore')
# np.seterr(divide='ignore', invalid='ignore')


class Hyperheuristic():
    def __init__(self, heuristic_space, problem, parameters=None, file_label=''):
        # Read the heuristic space
        if isinstance(heuristic_space, list):
            self.heuristic_space = heuristic_space
        elif isinstance(heuristic_space, str):
            with open('collections/' + heuristic_space,
                      'r') as operators_file:
                self.heuristic_space = [
                    eval(line.rstrip('\n')) for line in operators_file]
        else:
            raise HyperheuristicError("Invalid heuristic_space")

        # Assign default values
        if parameters is None:
            parameters = dict(cardinality=2,        # Search operators in MHs,  lvl:0
                              num_iterations=100,   # Iterations a MH performs, lvl:0
                              num_agents=30,        # Agents in population,     lvl:0
                              num_replicas=100,     # Replicas per each MH,     lvl:1
                              num_trials=100,       # Trials per HH step,       lvl:2
                              max_temperature=1e8,  # Initial temperature (SA), lvl:2
                              min_temperature=1e-8,  # Threshold temp. (SA),    lvl:2
                              cooling_rate=0.05)    # Cooling rate (SA),        lvl:2

        # Read the heuristic space (mandatory)
        self.problem = problem
        self.num_operators = len(self.heuristic_space)

        # Initialise other parameters
        self.parameters = parameters
        self.file_label = file_label

        # Read cardinality
        if isinstance(parameters['cardinality'], int):
            # Fixed cardinality
            self.cardinality_boundaries = [parameters['cardinality']] * 2
        elif isinstance(parameters['cardinality'], list):
            # Variable cardinality
            if len(parameters['cardinality']) == 1:
                self.cardinality_boundaries = [1, parameters['cardinality'][0]]
            elif len(parameters['cardinality']) == 2:
                self.cardinality_boundaries = parameters['cardinality'].sort()
            else:
                raise HyperheuristicError('Invalid cardinality!')

    def run(self):
        """
        Run Simulated Annealing to find the best metaheuristic.
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
        # Create the initial solution (it is fixed to be always Random Search)
        # It is 0 because operators._build_operators put random search as the
        # first operator.
        cardinality = 1
        encoded_solution = [0]
        solution = [self.heuristic_space[index] for index in encoded_solution]

        # Evaluate this solution
        performance, details = self.evaluate_metaheuristic(solution)

        # Initialise historical register
        historicals = dict(
            encoded_solution=encoded_solution,
            solution=solution,
            performance=performance,
            details=details)

        # Save this historical register
        _save_iteration(0, historicals, self.file_label)

        # Initialise the temperature and step counter
        temperature = self.parameters['max_temperature']
        step = 0

        # Print the first status update
        # print('{} - perf: {}, sol: {}'.format(step, performance, encoded_solution))

        # Perform the annealing simulation
        while temperature > self.parameters['min_temperature']:
            # Start trials
            for trial in range(self.parameters['num_trials']):

                # Generate a neighbour cardinality
                cardinality += np.random.randint(-1, 1)
                if cardinality < self.cardinality_boundaries[0]:
                    cardinality = self.cardinality_boundaries[0]
                elif cardinality > self.cardinality_boundaries[1]:
                    cardinality = self.cardinality_boundaries[1]

                # Generate a neighbour solution
                encoded_candidate_solution = np.random.randint(0, self.num_operators, cardinality)
                candidate_solution = [self.heuristic_space[index]
                                      for index in encoded_candidate_solution]

                # Evaluate this candidate solution
                candidate_performance, candidate_details =\
                    self.evaluate_metaheuristic(candidate_solution)

                # Determine the energy (performance) change
                delta_energy = candidate_performance - performance

                # Check improvement (Metropolis criterion)
                if (delta_energy < 0) or (np.random.rand() < np.exp(-delta_energy/temperature)):
                    encoded_solution = encoded_candidate_solution
                    solution = candidate_solution
                    performance = candidate_performance
                    details = candidate_details

                    # Save this historical register and break
                    step += 1
                    _save_iteration(step, {
                        'encoded_solution': encoded_solution,
                        'solution': solution,
                        'performance': performance,
                        'details': details},
                        self.file_label)

                    # print('{} - perf: {}, sol: {}'.format(step, performance, encoded_solution))

                    # When zero performance is reached end the simulation
                    if performance == 0.0:
                        return solution, performance, encoded_solution, historicals

            # Update temperature
            temperature *= 1 - self.parameters['cooling_rate']

        # Return the best solution found and its details
        return solution, performance, encoded_solution, historicals

    def evaluate_metaheuristic(self, search_operators):
        # Initialise the historical registers
        historical_data = list()
        fitness_data = list()
        position_data = list()

        # Run the metaheuristic several times
        for rep in range(1, self.parameters['num_replicas'] + 1):

            # Call the metaheuristic
            mh = Metaheuristic(self.problem, search_operators,
                               self.parameters['num_agents'],
                               self.parameters['num_iterations'])

            # Run this metaheuristic
            mh.run()

            # Store the historical values from this run
            historical_data.append(mh.historical)

            # Read and store the solution obtained
            _temporal_position, _temporal_fitness = mh.get_solution()
            fitness_data.append(_temporal_fitness)
            position_data.append(_temporal_position)

            # print('-- MH: {}, fitness={}'.format(rep, _temporal_fitness))

        # Determine a performance metric once finish the repetitions
        fitness_stats = self.get_statistics(fitness_data)

        return self.get_performance(fitness_stats), dict(
            historical=historical_data, fitness=fitness_data,
            positions=position_data, statistics=fitness_stats)

    def brute_force(self):
        # Apply all the search operators in the collection as 1-size MHs
        for operator_id in range(self.num_operators):
            # Read the corresponding operator
            operator = [self.heuristic_space[operator_id]]

            # Evaluate it within the metaheuristic structure
            operator_performance, operator_details = \
                self.evaluate_metaheuristic(operator)

            # Save information
            _save_iteration(operator_id, {
                'encoded_solution': operator,
                'performance': operator_performance,
                'details': operator_details},
                            self.file_label)

            # print('{}/{} - perf: {}'.format(operator_id + 1,
            #                                 self.num_operators,
            #                                 operator_performance))

    @staticmethod
    def get_performance(statistics):
        # Score function using the statistics from fitness values
        # perf = statistics['Med']  # Option 1
        # perf = statistics['Avg'] + statistics['Std']  # Option 2
        perf = statistics['Med'] + statistics['IQR']  # Option 3
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


def _save_iteration(iteration_number, variable_to_save, prefix=''):
    # Get the current date
    now = datetime.now()

    # Define the folder name
    sep = '-' if (prefix != '') else ''
    folder_name = "data_files/raw/" + prefix + sep + now.strftime("%m_%d_%Y")

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
