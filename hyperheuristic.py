# -*- coding: utf-8 -*-
"""
This module contains the Hyperheuristic class.

Created on Thu Jan  9 15:36:43 2020

@author: Jorge Mario Cruz-Duarte (jcrvz.github.io), e-mail: jorge.cruz@tec.mx
"""
import numpy as np
import scipy.stats as st
from metaheuristic import Metaheuristic
from datetime import datetime
import json
import tools as jt
from os.path import exists as _check_path
from os import makedirs as _create_path


class Hyperheuristic():
    """
    This is the Hyperheuristic class, each object corresponds to a hyper-heuristic process implemented with a heuristic collection from Operators to build metaheuristics using the Metaheuristic module.
    """

    def __init__(self, heuristic_space='default.txt', problem=None, parameters=None, file_label='', weights_array=None):
        """
        Create a hyper-heuristic process using a operator collection as heuristic space.

        :param str heuristic_space: Optional.
            The heuristic space or search space collection. It could be a string indicating the file name, assuming it
            is located in the folder ``./collections/``, or a list with tuples (check the default collection
            ``./collections/default.txt'``) just like ``operators.build_operators`` generates. The default is
            'default.txt'.
        :param dict problem:
            This is a dictionary containing the 'function' that maps a 1-by-D array of real values ​​to a real value,
            'is_constrained' flag that indicates the solution is inside the search space, and the 'boundaries' (a tuple
            with two lists of size D). These two lists correspond to the lower and upper limits of domain, such as:
            ``boundaries = (lower_boundaries, upper_boundaries)``

            **Note:** Dimensions (D) of search domain are read from these boundaries. The problem can be obtained from
            the ``benchmark_func`` module.
        :param dict parameters:
            Parameters to implement the hyper-heuristic procedure, the following fields must be provided: 'cardinality',
            'num_iterations', 'num_agents', 'num_replicas', 'num_steps', 'stagnation_percentage', 'max_temperature', and
            'cooling_rate'. The default is showing next:
                parameters = {cardinality=2,                # Max. numb. of SOs in MHs, lvl:1
                              num_iterations=100,           # Iterations a MH performs, lvl:1
                              num_agents=30,                # Agents in population,     lvl:1
                              num_replicas=100,             # Replicas per each MH,     lvl:2
                              num_steps=100,                # Trials per HH step,       lvl:2
                              stagnation_percentage=0.2,    # Stagnation,               lvl:2
                              max_temperature=100,          # Initial temperature (SA), lvl:2
                              cooling_rate=0.05}            # Cooling rate (SA),        lvl:2

            **Note:** Level (lvl) flag corresponds to the heuristic level of the parameter. lvl:1 concerns to mid-level
            heuristics like metaheuristics, and lvl:2 to high-level heuristics like hyper-heuristics.
        :param str file_label: Optional.
            Tag or label for saving files. The default is ''.
        :param numpy.array weights_array: Optional.
            Weights of the search operators, if there is a-priori information about them. The default is None.
        """
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
                              stagnation_percentage=0.2,  # Stagnation,         lvl:2
                              max_temperature=100,  # Initial temperature (SA), lvl:2
                              cooling_rate=0.05)    # Cooling rate (SA),        lvl:2

        # Read the problem
        if problem:
            self.problem = problem
        else:
            raise HyperheuristicError('Problem must be provided')

        # Read the heuristic space size
        self.num_operators = len(self.heuristic_space)

        # Read the weights (if it is entered)
        self.weights_array = weights_array

        # Initialise other parameters
        self.parameters = parameters
        self.file_label = file_label

    def run(self):
        """
        Run the hyper-heuristic based on Simulated Annealing (SA) to find the best metaheuristic. Each meatheuristic is
        run 'num_replicas' times to obtain statistics and then its performance. Once the process ends, it returns:
            - solution: The sequence of search operators that compose the metaheuristic.
            - performance: The metric value defined in ``get_performance``.
            - encoded_solution: The sequence of indices that correspond to the search operators.
            - historicals: A dictionary of information from each step. Its keys are: 'step', 'encoded_solution',
            'solution', 'performances', and 'details'. The latter, 'details', is also a dictionary which contains
            information about each replica carried out with the metaheuristic. Its fields are 'historical' (each
            iteration that the metaheuristic has performed), 'fitness', 'positions', and 'statistics'.

        :returns: solution (list), performance (float), encoded_solution (list), historicals (dict)
           solution : list
        """
        # Read the cardinality (which is the maximum allowed one)
        max_cardinality = self.parameters['cardinality']

        def obtain_neighbour_solution(sol=None):
            """
            This method selects a neighbour candidate solution for a given candidate solution ``sol``. To do so, it
            adds, deletes, or perturbate a randomly chosen operator index from the current sequence. If this sequence
            is None, the method returns a new 1-cardinality sequence at random.

            :param list sol: Optional.
                Sequence of heuristic indices (or encoded solution). The default is None, which means that there is no
                current sequence, so an initial one is required.

            :return: list.
            """
            if sol is None:
                # Create a new 1-MH from scratch by using a weights array (if so)
                encoded_neighbour = np.random.choice(self.num_operators, 1, replace=False, p=self.weights_array)
            elif isinstance(sol, np.ndarray):
                current_cardinality = len(sol)

                # First read the available actions. Those could be 'Add', 'Del', an 'Per'
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
                    new_operator = np.random.choice(np.setdiff1d(np.arange(self.num_operators), sol))

                    # Select where to add such an operator, since ``operator_location`` value represents:
                    #       0 - left side of the first operator
                    #       1 - right side of the first operator or left side of the second one,
                    #       ..., and so forth.
                    #
                    #       | operator 1 | operator 2 | operator 3 |     ...      |  operator N  |
                    #       0 <--------> 1 <--------> 2 <--------> 3 <-- ... --> N-1 <---------> N
                    operator_location = np.random.randint(current_cardinality + 1)

                    # Add the selected operator
                    encoded_neighbour = np.array((*sol[:operator_location], new_operator, *sol[operator_location:]))
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
                raise HyperheuristicError('Invalid type of current solution!')

            # Decode the neighbour solution
            neighbour = [self.heuristic_space[index] for index in encoded_neighbour]

            # Return the neighbour sequence and its decoded equivalent
            return encoded_neighbour, neighbour

        def obtain_temperature(step_val, function='boltzmann'):
            """
            Return the updated temperature according to a defined scheme ``function``.

            :param int step_val:
                Step (or iteration) value of the current state of the hyper-heuristic search.
            :param str function: Optional.
                Mechanism for updating the temperature. It can be 'exponential', 'fast', or 'boltzmann'. The default
                is 'boltzmann'.
            :return: float
            """
            if function == 'exponential':
                return self.parameters['max_temperature'] * np.power(
                    1 - self.parameters['cooling_rate'], step_val)
            elif function == 'fast':
                return self.parameters['max_temperature'] / step_val
            else:  # boltzmann
                return self.parameters['max_temperature'] / np.log(step_val + 1)

        # Acceptance function
        def check_acceptance(delta, temp, function='exponential'):
            """
            Return a flag indicating if the current performance value can be accepted according to the ``function``.

            :param float delta:
                Energy change for determining the acceptance probability.

            :param float temp:
                Temperature value for determining the acceptance probability.

            :param str function: Optional.
                Function for determining the acceptance proability. It can be 'exponential' or 'boltzmann'. The default
                is 'boltzmann'.

            :return: bool
            """
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
        historicals = dict(encoded_solution=best_encoded_solution, performance=best_performance,
                           details=current_details)

        # Save this historical register, step = 0
        _save_step(0, historicals, self.file_label)

        # Print the first status update, step = 0
        print('{} :: Step: {}, Perf: {}, e-Sol: {}'.format(self.file_label, 0, best_performance, best_encoded_solution))

        # Step, stagnation counter and its maximum value
        step = 0
        stag_counter = 0
        max_stag = round(self.parameters['stagnation_percentage'] * self.parameters['num_steps'])

        # Perform the annealing simulation as hyper-heuristic process
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
                _save_step(step, {
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
        """
        Evaluate the current sequence of ``search_operators`` as a metaheuristic. This process is repeated
        ``parameters['num_replicas']`` times and, then, the performance is determined. In the end, the method returns
        the performance value and the details for all the runs. These details are ``historical_data``, ``fitness_data``,
        ``position_data``, and ``fitness_stats``.

        :param list search_operators:
            Sequence of search operators. These must be in the tuple form (decoded version). Check the ``metaheuristic``
            module for further information.
        :return: float, dict
        """
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

        # Return the performance value and the corresponding details
        return self.get_performance(fitness_stats), dict(historical=historical_data, fitness=fitness_data,
            positions=position_data, statistics=fitness_stats)

    def brute_force(self):
        """
        This method performs a brute force procedure solving the problem via all the available search operators without
        integrating a high-level search method. So, each search operator is used as a 1-cardinality metaheuristic.
        Results are directly saved as json files

        :return: None.
        """
        # Apply all the search operators in the collection as 1-cardinality MHs
        for operator_id in range(self.num_operators):
            # Read the corresponding operator
            operator = [self.heuristic_space[operator_id]]

            # Evaluate it within the metaheuristic structure
            operator_performance, operator_details = self.evaluate_metaheuristic(operator)

            # Save information
            _save_step(operator_id, {
                'encoded_solution': operator_id,
                'performance': operator_performance,
                'statistics': operator_details['statistics']
            }, self.file_label)

            # print('{}/{} - perf: {}'.format(operator_id + 1, self.num_operators, operator_performance))

    def basic_metaheuristics(self, label):
        """
        This method performs a brute force procedure solving the problem via all the predefined metaheuristics in
        './collections/basicmetaheuristics.txt'. Many of them are 1-cardinality MHs but other are 2-cardinality ones.
        This process does not require a high-level search method. Results are directly saved as json files.

        :return: None.
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
            _save_step(operator_id, {
                'encoded_solution': operator_id,
                'performance': operator_performance,
                'statistics': operator_details['statistics']
            }, self.file_label)

            print('{} :: {}/{} - perf: {}'.format(label, operator_id + 1, self.num_operators, operator_performance))

    @staticmethod
    def get_performance(statistics):
        """
        Return the performance from fitness values obtained from running a metaheuristic several times. This method uses
        the Median and Interquartile Range values for such a purpose:
            performance = Med{fitness values} + IQR{fitness values}
        **Note:** If an alternative formula is needed, check the commented options.
        :param statistics:
        :type statistics:
        :return:
        :rtype:
        """
        # TODO: Verify if using conditional for choosing between options is not cost computing
        # return statistics['Med']                                                                  # Option 1
        # return statistics['Avg'] + statistics['Std']                                              # Option 2
        return statistics['Med'] + statistics['IQR']                                                # Option 3
        # return statistics['Avg'] + statistics['Std'] + statistics['Med'] + statistics['IQR']      # Option 4

    @staticmethod
    def get_statistics(raw_data):
        """
        Return statistics from all the fitness values found after running a metaheuristic several times. The oncoming
        statistics are ``nob`` (number of observations), ``Min`` (minimum), ``Max`` (maximum), ``Avg`` (average),
        ``Std`` (standard deviation), ``Skw`` (skewness), ``Kur`` (kurtosis), ``IQR`` (interquartile range),
        ``Med`` (median), and ``MAD`` (Median absolute deviation).

        :param list raw_data:
            List of the fitness values.
        :return: dict
        """
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


# %% ADDITIONAL TOOLS

def _save_step(step_number, variable_to_save, prefix=''):
    """
    This method saves all the information corresponding to specific step.

    :param int step_number:
        Value of the current step in the hyper-heuristic procedure. If it is not a hyper-heuristic, this integer
        corresponds to operator index.
    :param dict variable_to_save:
        Variables to save in dictionary format.
    :param str prefix: Optional.
        Additional information to be used in naming the file. The default is ''.
    :return:
    :rtype:
    """
    # Get the current date
    now = datetime.now()

    # Define the folder name
    sep = '-' if (prefix != '') else ''
    folder_name = "data_files/raw/" + prefix + sep + now.strftime("%m_%d_%Y")

    # Check if this path exists
    if not _check_path(folder_name):
        _create_path(folder_name)

    # Create a new file for this step
    with open(folder_name + f"/{step_number}-" + now.strftime(
            "%H_%M_%S") + ".json", 'w') as json_file:
        json.dump(variable_to_save, json_file, cls=jt.NumpyEncoder)


class HyperheuristicError(Exception):
    """
    Simple HyperheuristicError to manage exceptions.
    """
    pass

