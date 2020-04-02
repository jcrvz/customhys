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
    def __init__(self,
                 heuristic_space='default.txt',
                 problem=None,
                 parameters=None,
                 file_label='',
                 weights_array=None):

        # Read the heuristic space
        if isinstance(heuristic_space, list):
            self.heuristic_space = heuristic_space
        elif isinstance(heuristic_space, str):
            with open('collections/' + heuristic_space, 'r') as operators_file:
                self.heuristic_space = [eval(line.rstrip('\n')) for line in operators_file]
        else:
            raise HyperheuristicError("Invalid heuristic_space")

        # Assign default values
        if parameters is None:
            parameters = dict(cardinality=2,        # Max. numb. of SOs in MHs, lvl:0
                              num_iterations=100,   # Iterations a MH performs, lvl:0
                              num_agents=30,        # Agents in population,     lvl:0
                              num_replicas=100,     # Replicas per each MH,     lvl:1
                              num_steps=100,        # Trials per HH step,       lvl:2
                              stagnation_percentage= 0.2, # Stagnation,         lvl:2
                              max_temperature=100,  # Initial temperature (SA), lvl:2
                              cooling_rate=0.05)    # Cooling rate (SA),        lvl:2

        # Read the heuristic space (mandatory)
        self.problem = problem
        self.num_operators = len(self.heuristic_space)
        self.weights_array = weights_array

        # Initialise other parameters
        self.parameters = parameters
        self.file_label = file_label

        # Read cardinality
        # if isinstance(parameters['cardinality'], int):
        #     # Fixed cardinality
        #     self.cardinality_boundaries = [parameters['cardinality']]
        # elif isinstance(parameters['cardinality'], list):
        #     # Variable cardinality
        #     if len(parameters['cardinality']) == 1:
        #         self.cardinality_boundaries = [1, parameters['cardinality'][0]]
        #     elif len(parameters['cardinality']) == 2:
        #         self.cardinality_boundaries = parameters['cardinality'].sort()
        #     else:
        #         raise HyperheuristicError('Invalid cardinality!')

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
        # Read the cardinality (which is the maximum allowed one)
        max_cardinality = self.parameters['cardinality']

        # Mechanism to mutate/perturbate the current solution
        def obtain_neighbour_solution(sol=None):
            if sol is None:
                # Create a new 1-MH from scratch by using a weights array (if so)
                encoded_neighbour = np.random.choice(self.num_operators, 1, replace=False, p=self.weights_array)
            elif isinstance(sol, np.ndarray):
                current_cardinality = len(sol)

                # First read the available actions 'Add', 'Del',
                if current_cardinality >= max_cardinality:
                    available_options = ['Del', 'Per']
                elif current_cardinality <= 1:
                    available_options = ['Add', 'Per']
                else:
                    available_options = ['Add', 'Del', 'Per']

                # Decide (randomly) which action to do
                action = np.random.choice(available_options)

                # Perform the corresponding action
                if action == 'Add':
                    # Select an operator excluding the ones in the current solution
                    new_operator = np.random.choice(
                        np.setdiff1d(np.arange(self.num_operators), sol))

                    # Select where to add such an operator
                    #       since   0 - left side of the first operator
                    #               1 - right side of the first operator or left side of the second one
                    #               ..., and so forth
                    operator_location = np.random.randint(current_cardinality + 1)

                    # Add the selected operator
                    encoded_neighbour = np.array((*sol[:operator_location],
                                                  new_operator,
                                                  *sol[operator_location:]))
                elif action == 'Del':
                    # Delete an operator randomly selected
                    encoded_neighbour = np.delete(sol, np.random.randint(current_cardinality))
                else:
                    # Copy the current solution
                    encoded_neighbour = np.copy(sol)

                    # Perturbate an operator randomly selected excluding the existing ones
                    encoded_neighbour[np.random.randint(current_cardinality)] = np.random.choice(
                        np.setdiff1d(np.arange(self.num_operators), sol))
            else:
                raise HyperheuristicError("Invalid type of current solution!")

            # Decode the neighbour solution
            neighbour = [self.heuristic_space[index] for index in encoded_neighbour]

            return encoded_neighbour, neighbour

        # Define the temperature update function
        def obtain_temperature(step_val, function='boltzmann'):
            if function == 'exponential':
                return self.parameters['max_temperature'] * np.power(
                    1 - self.parameters['cooling_rate'], step_val)
            elif function == 'fast':
                return self.parameters['max_temperature'] / step_val
            else:  # boltzmann
                return self.parameters['max_temperature'] / np.log(step_val + 1)

        # Acceptance function
        def check_acceptance(delta, temp, function='exponential'):
            if function == 'exponential':
                return (delta_energy <= 0) or (np.random.rand() < np.exp(-delta / temp))
            else:  # boltzmann
                return (delta_energy <= 0) or (np.random.rand() < 1. / (1. + np.exp(delta / temp)))

        # Create the initial solution
        current_encoded_solution, current_solution = obtain_neighbour_solution()

        # Evaluate this solution
        current_performance, current_details = self.evaluate_metaheuristic(current_solution)

        # Initialise the best solution and its performance
        best_encoded_solution = np.copy(current_encoded_solution)
        best_performance = current_performance

        # Initialise historical register
        historicals = dict(
            encoded_solution=best_encoded_solution,
            # solution=solution, (it is to save some space)
            performance=best_performance,
            details=current_details)

        # Save this historical register, step = 0
        _save_iteration(0, historicals, self.file_label)

        # Print the first status update, step = 0
        print('{} :: Step: {}, Perf: {}, e-Sol: {}'.format(
            self.file_label, 0, best_performance, best_encoded_solution))

        # Step, stagnation counter and its maximum value
        step = 0
        stag_counter = 0
        max_stag = round(self.parameters['stagnation_percentage'] * self.parameters['num_steps'])

        # Perform the annealing simulation
        while (step <= self.parameters['num_steps']) and (stag_counter <= max_stag):
            step += 1

            # Generate a neighbour solution (just indices-codes)
            candidate_encoded_solution, candidate_solution = obtain_neighbour_solution(current_encoded_solution)

            # Evaluate this candidate solution
            candidate_performance, candidate_details = self.evaluate_metaheuristic(candidate_solution)

            # Determine the energy (performance) change
            delta_energy = candidate_performance - current_performance

            # Update temperature
            temperature = obtain_temperature(step)

            # Accept the current solution via Metropolis criterion
            if check_acceptance(delta_energy, temperature):
                # Update the current solution and its performance
                current_encoded_solution = np.copy(candidate_encoded_solution)
                current_performance = candidate_performance
                # if delta_energy > 0:
                #     print('{} :: Step: {}, Perf: {}, e-Sol: {} [Accepted]'.format(
                #         self.file_label, step, current_performance, current_encoded_solution))

            # If the candidate solution is better or equal than the current best solution
            if candidate_performance < best_performance:
                # Update the best solution and its performance
                best_encoded_solution = np.copy(candidate_encoded_solution)
                best_performance = candidate_performance

                # Reset the stagnation counter
                stag_counter = 0

                # Save this information
                _save_iteration(step, {
                    'encoded_solution': best_encoded_solution,
                    'performance': best_performance,
                    'details': candidate_details
                }, self.file_label)

                # Print update
                print('{} :: Step: {}, Perf: {}, e-Sol: {}'.format(
                    self.file_label, step, best_performance, best_encoded_solution))
            else:
                # Update the stagnation counter
                stag_counter += 1

        # Return the best solution found and its details
        return current_solution, current_performance, current_encoded_solution, historicals

    def evaluate_metaheuristic(self, search_operators):
        # Initialise the historical registers
        historical_data = list()
        fitness_data = list()
        position_data = list()

        # Run the metaheuristic several times
        for rep in range(1, self.parameters['num_replicas'] + 1):

            # Call the metaheuristic
            mh = Metaheuristic(self.problem, search_operators, self.parameters['num_agents'],
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
        """
        This method performs a brute force procedure solving all the problem via all the available search
        operators without integrating a high-level search method.

        :return: Nothing

        It save results as json files.
        """
        # Apply all the search operators in the collection as 1-size MHs
        for operator_id in range(self.num_operators):
            # Read the corresponding operator
            operator = [self.heuristic_space[operator_id]]

            # Evaluate it within the metaheuristic structure
            operator_performance, operator_details = self.evaluate_metaheuristic(operator)

            # Save information
            _save_iteration(operator_id, {
                'encoded_solution': operator_id,
                'performance': operator_performance,
                'statistics': operator_details['statistics']
            }, self.file_label)

            # print('{}/{} - perf: {}'.format(operator_id + 1,
            #                                 self.num_operators,
            #                                 operator_performance))


    def basic_metaheuristics(self):
        """
        This method performs a brute force procedure solving all the problem via all the available search
        operators without integrating a high-level search method.

        :return: Nothing

        It save results as json files.
        """
        # Apply all the search operators in the collection as 1-size MHs
        for operator_id in range(self.num_operators):
            operator = self.heuristic_space[operator_id]
            # Read the corresponding operator

            if isinstance(operator, tuple):
                operator = [operator]

            # Evaluate it within the metaheuristic structure
            operator_performance, operator_details = self.evaluate_metaheuristic(operator)

            # Save information
            _save_iteration(operator_id, {
                'encoded_solution': operator_id,
                'performance': operator_performance,
                'statistics': operator_details['statistics']
            }, self.file_label)

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


def set_problem(function, boundaries, is_constrained=True, weights=None):
    return {'function': function, 'boundaries': boundaries,
            'is_constrained': is_constrained, 'weights': weights}


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
