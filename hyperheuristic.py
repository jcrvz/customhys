# -*- coding: utf-8 -*-
"""
This module contains the Hyperheuristic class.

Created on Thu Jan  9 15:36:43 2020

@author: Jorge Mario Cruz-Duarte (jcrvz.github.io), e-mail: jorge.cruz@tec.mx
"""

import numpy as np
import random
from itertools import product
import pandas as pd
import scipy.stats as st
from metaheuristic import Metaheuristic
from datetime import datetime
import operators as Operators
import json
import tools as jt
from os.path import exists as _check_path
from os import makedirs as _create_path
from deprecated import deprecated
import tensorflow as tf
import keras
from machine_learning import create_autoencoder as _create_autoencoder


class Hyperheuristic:
    """
    This is the Hyperheuristic class, each object corresponds to a hyper-heuristic process implemented with a heuristic
    collection from Operators to build metaheuristics using the Metaheuristic module.
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
            This is a dictionary containing the 'function' that maps a 1-by-D array of real values to a real value,
            'is_constrained' flag that indicates the solution is inside the search space, and the 'boundaries' (a tuple
            with two lists of size D). These two lists correspond to the lower and upper limits of domain, such as:
            ``boundaries = (lower_boundaries, upper_boundaries)``

            **Note:** Dimensions (D) of search domain are read from these boundaries. The problem can be obtained from
            the ``benchmark_func`` module.
        :param dict parameters:
            Parameters to implement the hyper-heuristic procedure, the following fields must be provided: 'cardinality',
            'num_iterations', 'num_agents', 'num_replicas', 'num_steps', 'stagnation_percentage', 'max_temperature', and
            'cooling_rate'. The default is showing next:
                parameters = {cardinality=3,                # Max. numb. of SOs in MHs, lvl:1
                              num_iterations=100,           # Iterations a MH performs, lvl:1
                              num_agents=30,                # Agents in population,     lvl:1
                              num_replicas=50,              # Replicas per each MH,     lvl:2
                              num_steps=100,                # Trials per HH step,       lvl:2
                              stagnation_percentage=0.3,    # Stagnation percentage,    lvl:2
                              max_temperature=200,          # Initial temperature (SA), lvl:2
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
            self.heuristic_space_label = f'custom_heuristic_space_len_{len(heuristic_space)}'
            self.heuristic_space = heuristic_space
        elif isinstance(heuristic_space, str):
            self.heuristic_space_label = heuristic_space[:heuristic_space.rfind('.')]
            with open('collections/' + heuristic_space, 'r') as operators_file:
                self.heuristic_space = [eval(line.rstrip('\n')) for line in operators_file]
        else:
            raise HyperheuristicError('Invalid heuristic_space')

        # Assign default values
        if not parameters:
            parameters = dict(cardinality=3,  # Max. numb. of SOs in MHs, lvl:1
                              cardinality_min=1,  # Min. numb. of SOs in MHs, lvl:1
                              num_iterations=100,  # Iterations a MH performs, lvl:1
                              num_agents=30,  # Agents in population,     lvl:1
                              as_mh=False,  # HH sequence as a MH?,     lvl:2
                              num_replicas=50,  # Replicas per each MH,     lvl:2
                              num_steps=200,  # Trials per HH step,       lvl:2
                              stagnation_percentage=0.37,  # Stagnation percentage,    lvl:2
                              max_temperature=1,  # Initial temperature (SA), lvl:2
                              min_temperature=1e-6,  # Min temperature (SA),     lvl:2
                              cooling_rate=1e-3,  # Cooling rate (SA),        lvl:2
                              temperature_scheme='fast',  # Temperature updating (SA),lvl:2 *new
                              acceptance_scheme='exponential',  # Acceptance mode,          lvl:2 *new
                              allow_weight_matrix=True,  # Weight matrix,            lvl:2 *new
                              trial_overflow=False,  # Trial overflow policy,    lvl:2 *new
                              learnt_dataset=None,  # If it is a learnt dataset related with the heuristic space
                              repeat_operators=True,  # Allow repeating SOs inSeq,lvl:2
                              verbose=True)  # Verbose process,          lvl:2
        # Read the problem
        if problem:
            self.problem = problem
        else:
            raise HyperheuristicError('Problem must be provided')

        # Read the heuristic space size and create the active set
        self.num_operators = len(self.heuristic_space)
        self.current_space = np.arange(self.num_operators)

        # Read the weights (if it is entered)
        self.weights = weights_array
        self.transition_matrix = None

        # Initialise other parameters
        self.parameters = parameters
        self.file_label = file_label

        self.max_cardinality = None
        self.min_cardinality = None
        self.num_iterations = None
        self.toggle_seq_as_meta(parameters['as_mh'])
        self.autoencoder = None

    def toggle_seq_as_meta(self, as_mh=None):
        if as_mh is None:
            self.parameters['as_mh'] = not self.parameters['as_mh']
            self.toggle_seq_as_meta(self.parameters['as_mh'])
        else:
            if as_mh:
                self.max_cardinality = self.parameters['cardinality']
                self.min_cardinality = self.parameters['cardinality_min']
                self.num_iterations = self.parameters['num_iterations']
            else:
                self.max_cardinality = self.parameters['num_iterations']
                self.min_cardinality = self.parameters['cardinality_min']
                self.num_iterations = 1

    def _choose_action(self, current_cardinality, previous_action=None, available_options=None):
        # First read the available actions. Those could be ...
        if available_options is None:
            available_options = ['Add', 'AddMany', 'Remove', 'RemoveMany', 'Shift', 'LocalShift', 'Swap', 'Restart',
                                 'Mirror', 'Roll', 'RollMany']

        # Black list (to avoid repeating the some actions in a row)
        if previous_action:
            if previous_action == 'Mirror':
                available_options.remove('Mirror')

        # Disregard those with respect to the current cardinality. It also considers the case of fixed cardinality
        if current_cardinality <= self.min_cardinality + 1:
            available_options.remove('RemoveMany')

            if current_cardinality <= self.min_cardinality:
                available_options.remove('Remove')

        if current_cardinality <= 1:
            available_options.remove('Swap')
            available_options.remove('Mirror')  # not an error, but to prevent wasting time

        if current_cardinality >= self.max_cardinality - 1:
            available_options.remove('AddMany')

            if current_cardinality >= self.max_cardinality:
                available_options.remove('Add')

        # Decide (randomly) which action to do
        return np.random.choice(available_options)

    def _obtain_candidate_solution(self, sol=None, action=None, operators_weights=None):
        """
        This method selects a new candidate solution for a given candidate solution ``sol``. To do so, it
        adds, deletes, or perturbate a randomly chosen operator index from the current sequence. If this sequence
        is None, the method returns a new 1-cardinality sequence at random.

        :param list|int sol: Optional.
            Sequence of heuristic indices (or encoded solution). If `sol` is an integer, it is assumed that this is
            the cardinality required for initial random sequence. The default is None, which means that there is no
            current sequence, so an initial one is required.

        :return: list.
        """
        # TODO: fix active set to all the possible candidate approaches

        # Create a new MH with min_cardinality from scratch by using a weights array (if so)
        # if action is given, it is assumed the way of obtaining this intial solution
        if sol is None:
            if action == 'max_frequency':
                # Each search operator per step corresponds to the most frequent one: uMH weight matrix is required
                # this option only works for transfer learning
                encoded_neighbour = [weights_per_step.argmax() for weights_per_step in operators_weights]

            else:
                initial_cardinality = self.min_cardinality if self.parameters['as_mh'] else \
                    (self.max_cardinality + self.min_cardinality) // 2

                operators_weights = operators_weights if operators_weights else self.weights

                encoded_neighbour = np.random.choice(
                    self.current_space if (operators_weights is None) else self.num_operators, initial_cardinality,
                    replace=self.parameters['repeat_operators'], p=operators_weights)

        # If sol is an integer, assume that it refers to the cardinality
        elif isinstance(sol, int):
            operators_weights = self.weights if operators_weights is None else operators_weights

            encoded_neighbour = np.random.choice(
                self.current_space if (operators_weights is None) else self.num_operators, sol,
                replace=self.parameters['repeat_operators'], p=operators_weights)

        elif isinstance(sol, (np.ndarray, list)):
            # TODO: fix to consider the inputted weight array
            sol = np.array(sol)
            current_cardinality = len(sol)

            # Choose (randomly) which action to do
            if not action:
                action = self._choose_action(current_cardinality)

            # print(action)

            # Perform the corresponding action
            if (action == 'Add') and (current_cardinality < self.max_cardinality):
                # Select an operator excluding the ones in the current solution
                selected_operator = np.random.choice(np.setdiff1d(self.current_space, sol)
                                                     if not self.parameters['repeat_operators'] else self.current_space)

                # Select where to add such an operator, since ``operator_location`` value represents:
                #       0 - left side of the first operator
                #       1 - right side of the first operator or left side of the second one,
                #       ..., and so forth.
                #
                #       | operator 1 | operator 2 | operator 3 |     ...      |  operator N  |
                #       0 <--------> 1 <--------> 2 <--------> 3 <-- ... --> N-1 <---------> N
                operator_location = np.random.randint(current_cardinality + 1)

                # Add the selected operator
                encoded_neighbour = np.array((*sol[:operator_location], selected_operator, *sol[operator_location:]))

            elif (action == 'AddMany') and (current_cardinality < self.max_cardinality - 1):
                encoded_neighbour = np.copy(sol)
                for _ in range(np.random.randint(1, self.max_cardinality - current_cardinality + 1)):
                    encoded_neighbour = self._obtain_candidate_solution(sol=encoded_neighbour, action='Add')

            elif (action == 'Remove') and (current_cardinality > self.min_cardinality):
                # Delete an operator randomly selected
                encoded_neighbour = np.delete(sol, np.random.randint(current_cardinality))

            elif (action == 'RemoveMany') and (current_cardinality > self.min_cardinality + 1):
                encoded_neighbour = np.copy(sol)
                for _ in range(np.random.randint(1, current_cardinality - self.min_cardinality + 1)):
                    encoded_neighbour = self._obtain_candidate_solution(sol=encoded_neighbour, action='Remove')

            elif action == 'Shift':
                # Perturbate an operator randomly selected excluding the existing ones
                encoded_neighbour = np.copy(sol)
                encoded_neighbour[np.random.randint(current_cardinality)] = np.random.choice(
                    np.setdiff1d(self.current_space, sol)
                    if not self.parameters['repeat_operators'] else self.num_operators)

            elif action == 'LocalShift':  # It only works with the full set
                # Perturbate an operator randomly selected +/- 1 excluding the existing ones
                encoded_neighbour = np.copy(sol)
                operator_location = np.random.randint(current_cardinality)
                neighbour_direction = 1 if random.random() < 0.5 else -1  # +/- 1
                selected_operator = (encoded_neighbour[operator_location] + neighbour_direction) % self.num_operators

                # If repeat is true and the selected_operator is repeated, then apply +/- 1 until it is not repeated
                while (not self.parameters['repeat_operators']) and (selected_operator in encoded_neighbour):
                    selected_operator = (selected_operator + neighbour_direction) % self.num_operators

                encoded_neighbour[operator_location] = selected_operator

            elif (action == 'Swap') and (current_cardinality > 1):
                # Swap two elements randomly chosen
                if current_cardinality == 2:
                    encoded_neighbour = np.copy(sol)[::-1]

                elif current_cardinality > 2:
                    encoded_neighbour = np.copy(sol)
                    ind1, ind2 = np.random.choice(current_cardinality, 2, replace=False)
                    encoded_neighbour[ind1], encoded_neighbour[ind2] = encoded_neighbour[ind2], encoded_neighbour[ind1]

                else:
                    raise HyperheuristicError('Swap cannot be applied! current_cardinality < 2')

            elif action == 'Mirror':
                # Mirror the sequence of the encoded_neighbour
                encoded_neighbour = np.copy(sol)[::-1]

            elif action == 'Roll':
                # Move a step forward or backward, depending on a random variable, all the sequence
                encoded_neighbour = np.roll(sol, 1 if random.random() < 0.5 else -1)

            elif action == 'RollMany':
                # Move many (at random) steps forward or backward, depending on a random variable, all the sequence
                encoded_neighbour = np.roll(sol, np.random.randint(current_cardinality) * (
                    1 if random.random() < 0.5 else -1))

            elif action == 'Restart':
                # Restart the entire sequence
                encoded_neighbour = self._obtain_candidate_solution(current_cardinality)

            else:
                raise HyperheuristicError(f'Invalid action = {action} to perform!')
        else:
            raise HyperheuristicError('Invalid type of current solution!')

        # Decode the neighbour solution
        # neighbour = [self.heuristic_space[index] for index in encoded_neighbour]

        # Return the neighbour sequence and its decoded equivalent
        return encoded_neighbour

    def _obtain_temperature(self, step_val, function='boltzmann'):
        """
        Return the updated temperature according to a defined scheme ``function``.

        :param int step_val:
            Step (or iteration) value of the current state of the hyper-heuristic search.
        :param str function: Optional.
            Mechanism for updating the temperature. It can be 'exponential', 'fast', or 'boltzmann'. The default
            is 'boltzmann'.
        :return: float
        """
        if function == 'fast':
            return self.parameters['max_temperature'] / step_val

        elif function == 'linear':
            return self.parameters['max_temperature'] - (1 - self.parameters['cooling_rate']) * step_val

        elif function == 'quadratic':
            return self.parameters['max_temperature'] / (1 + (1 - self.parameters['cooling_rate']) * (step_val ** 2))

        elif function == 'logarithmic':
            return self.parameters['max_temperature'] / (
                    1 + (1 - self.parameters['cooling_rate']) * np.log(step_val + 1))

        elif function == 'exponential':
            return self.parameters['max_temperature'] * np.power(1 - self.parameters['cooling_rate'], step_val)

        elif function == 'boltzmann':
            return self.parameters['max_temperature'] / np.log(step_val + 1)

        else:
            raise HyperheuristicError('Invalid temperature scheme')

    def _check_acceptance(self, delta, acceptation_scheme, temp, energy_zero=1.0):
        """
        Return a flag indicating if the current performance value can be accepted according to ``acceptation_scheme``.

        :param float delta:
            Energy change for determining the acceptance probability.

        :param str acceptation_scheme: Optional.
            Function for determining the acceptance probability. It can be 'exponential', 'boltzmann', or 'greedy'. The
             default is 'greedy'.

        :param float temp: Required for acceptation_scheme = ('exponential'|'boltzmann')
            Temperature value for determining the acceptance probability.

        :param float energy_zero: Required for acceptation_scheme = ('exponential'|'boltzmann')
            Energy value to scale the temperature measurement. The default value is 1.

        :return: bool
        """

        if acceptation_scheme == 'exponential':
            probability = np.min([np.exp(-delta / (energy_zero * temp)), 1])
            if self.parameters['verbose']:
                print(', [Delta: {:.2e}, ArgProb: {:.2e}, Prob: {:.2f}]'.format(
                    delta, -delta / (energy_zero * temp), probability), end=' ')
            return np.random.rand() < probability
        elif acceptation_scheme == 'boltzmann':
            probability = 1. / (1. + np.exp(delta / temp))
            return (delta <= 0.0) or (np.random.rand() <= probability)
        else:  # Greedy
            return delta <= 0.0

    def __stagnation_check(self, stag_counter):
        return stag_counter > (self.parameters['stagnation_percentage'] * self.parameters['num_steps'])

    def _check_finalisation(self, step, stag_counter, *args):
        """ TODO: update this information and check if step should be here
        General finalisation approach implemented for the methodology working as a hyper-heuristic. It mainly depends on
        `step` (current iteration number) and `stag_counter` (current stagnation iteration number). There are other
         variables that can be considered such as `temperature`. These additional variables must be args[0] <= 0.0.
        """
        return (step > self.parameters['num_steps']) or (
                self.__stagnation_check(stag_counter) and not self.parameters['trial_overflow']) or \
               (any([var <= 0.0 for var in args]))

    # def _check_improvement(self, new_perf, best_perf, new_pos, best_pos):
    #     if self.parameters['as_mh']:
    #         return new_perf < best_perf
    #     else:
    #         return (new_perf <= best_perf) and (len(new_pos) <= len(best_pos))  # (new_perf < best_perf) or

    def get_operators(self, sequence):
        return [self.heuristic_space[index] for index in sequence]

    @deprecated(version='1.0.1', reason="Use solve instead")
    def run(self, temperature_scheme=None, acceptance_scheme=None):
        if temperature_scheme:
            self.parameters['temperature_scheme'] = temperature_scheme
        if acceptance_scheme:
            self.parameters['acceptance_scheme'] = acceptance_scheme
        return self._solve_static()

    def solve(self, mode='static', kw_parameters={}):
        if mode == 'dynamic':
            return self._solve_dynamic(kw_parameters)
        elif mode == 'neural_network':
            return self._solve_neural_network(kw_parameters)
        elif mode == 'transfer_learning':
            return self._solve_transfer_learning(kw_parameters)
        else:  # default: 'static'
            return self._solve_static()

    def _solve_static(self):
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

        :returns: solution (list), performance (float), encoded_solution (list)

        TODO: Generalise this code using different blocks, now, it is pretty close to SA
        """

        # %% INITIALISER PART

        # PERTURBATOR (GENERATOR): Create the initial solution
        current_solution = self._obtain_candidate_solution()

        # Evaluate this solution
        current_performance, current_details = self.evaluate_candidate_solution(current_solution)

        # Initialise some additional variables
        initial_energy = np.abs(current_performance) + 1  # np.copy(current_performance)
        historical_current = [current_performance]
        historical_best = [current_performance]

        # SELECTOR: Initialise the best solution and its performance
        best_solution = np.copy(current_solution)
        best_performance = current_performance

        # Save this historical register, step = 0
        _save_step(0, dict(encoded_solution=best_solution, performance=best_performance,
                           details=current_details), self.file_label)

        # Step, stagnation counter and its maximum value
        step = 0
        stag_counter = 0
        action = None
        temperature = self.parameters['max_temperature']

        # Print the first status update, step = 0
        if self.parameters['verbose']:
            print('{} :: Step: {:4d}, Action: {:12s}, Temp: {:.2e}, Card: {:3d}, Perf: {:.2e} [Initial]'.format(
                self.file_label, step, 'None', temperature, len(current_solution), current_performance))
            # ''.join([chr(97 + round(x * 25 / self.num_operators)) for x in current_solution])))

        # Perform a metaheuristic (now, Simulated Annealing) as hyper-heuristic process
        while not self._check_finalisation(step, stag_counter,
                                           temperature - self.parameters['min_temperature']):
            # Update step and temperature
            step += 1
            temperature = self._obtain_temperature(step, self.parameters['temperature_scheme'])

            # Generate a neighbour solution (just indices-codes)
            action = self._choose_action(len(current_solution), action)
            candidate_solution = self._obtain_candidate_solution(sol=current_solution, action=action)

            # Evaluate this candidate solution
            candidate_performance, candidate_details = self.evaluate_candidate_solution(candidate_solution)

            # Print update
            if self.parameters['verbose']:
                print('{} :: Step: {:4d}, Action: {:12s}, Temp: {:.2e}, Card: {:3d}, '.format(
                    self.file_label, step, action, temperature, len(candidate_solution)) +
                      'candPerf: {:.2e}, currPerf: {:.2e}, bestPerf: {:.2e}'.format(
                          candidate_performance, current_performance, best_performance), end=' ')

            # Accept the current solution using a given acceptance_scheme
            if self._check_acceptance(candidate_performance - current_performance, self.parameters['acceptance_scheme'],
                                      temperature, initial_energy):

                # Update the current solution and its performance
                current_solution = np.copy(candidate_solution)
                current_performance = candidate_performance

                # Add acceptance mark
                if self.parameters['verbose']:
                    print('A', end='')

            # If the candidate solution is better or equal than the current best solution
            if candidate_performance <= best_performance:

                # Update the best solution and its performance
                best_solution = np.copy(candidate_solution)
                best_performance = candidate_performance

                # Reset the stagnation counter
                stag_counter = 0

                # Save this information
                _save_step(step, {
                    'encoded_solution': best_solution,
                    'performance': best_performance,
                    'details': candidate_details
                }, self.file_label)

                # Add improvement mark
                if self.parameters['verbose']:
                    print('+', end='')
            else:
                # Update the stagnation counter
                stag_counter += 1

            historical_current.append(current_performance)
            historical_best.append(best_performance)
            # Add ending mark
            if self.parameters['verbose']:
                print('')

        # Print the best one
        if self.parameters['verbose']:
            print('\nBEST --> Perf: {}, e-Sol: {}'.format(best_performance, best_solution))

        # Return the best solution found and its details
        return best_solution, best_performance, historical_current, historical_best

    def _solve_transfer_learning(self, kw_weighting_params):
    
        # Check if there is a previous weight matrix stored
        if self.parameters['learnt_dataset']:
            weight_matrix = self._get_weight_matrix(
                category=self.problem['features'],
                values_dict={'Pop': self.parameters['num_agents'],
                             'Dim': self.problem['dimensions']},
                file_path=self.parameters['learnt_dataset'])
        else:
            raise HyperheuristicError("learnt_dataset is required for using this method")

        # Obtain an initial guess sequence by choosing the operators with the maximal likelihood for each step
        current_solution = self._obtain_candidate_solution(action="max_frequency", operators_weights=weight_matrix)

        # Evaluate this candidate solution
        current_performance, current_details = self.evaluate_candidate_solution(current_solution)

        # Update the current performance
        current_performance *= len(current_solution)

        # Initialise some additional variables
        initial_energy = np.abs(current_performance) + 1  # np.copy(current_performance)
        historical_current = [current_performance]
        historical_best = [current_performance]

        # SELECTOR: Initialise the best solution and its performance
        best_solution = np.copy(current_solution)
        best_performance = current_performance

        # Save this historical register, step = 0
        _save_step(0, dict(encoded_solution=best_solution, performance=best_performance,
                           details=current_details), self.file_label)

        # Step, stagnation counter and its maximum value
        step = 0
        stag_counter = 0
        action = None
        temperature = self.parameters['max_temperature']

        # Define the available actions to use in the process
        actions = ['Remove', 'RemoveMany', 'Shift', 'LocalShift', 'Swap', 'Roll', 'RollMany']

        # Print the first status update, step = 0
        if self.parameters['verbose']:
            print('{} :: Step: {:4d}, Action: {:12s}, Temp: {:.2e}, Card: {:3d}, Perf: {:.2e} [Initial]'.format(
                self.file_label, step, 'None', temperature, len(current_solution), current_performance))
            # ''.join([chr(97 + round(x * 25 / self.num_operators)) for x in current_solution])))

        # Perform a metaheuristic (now, Simulated Annealing) as hyper-heuristic process
        while not self._check_finalisation(step, stag_counter,
                                           temperature - self.parameters['min_temperature']):
            # Update step and temperature
            step += 1
            temperature = self._obtain_temperature(step, self.parameters['temperature_scheme'])

            # Generate a neighbour solution (just indices-codes)
            action = self._choose_action(len(current_solution), action)
            candidate_solution = self._obtain_candidate_solution(sol=current_solution, action=action)

            # Evaluate this candidate solution
            candidate_performance, candidate_details = self.evaluate_candidate_solution(candidate_solution)

            # Print update
            if self.parameters['verbose']:
                print('{} :: Step: {:4d}, Action: {:12s}, Temp: {:.2e}, Card: {:3d}, '.format(
                    self.file_label, step, action, temperature, len(candidate_solution)) +
                      'candPerf: {:.2e}, currPerf: {:.2e}, bestPerf: {:.2e}'.format(
                          candidate_performance, current_performance, best_performance), end=' ')

            # Accept the current solution using a given acceptance_scheme
            if self._check_acceptance(candidate_performance - current_performance, self.parameters['acceptance_scheme'],
                                      temperature, initial_energy):

                # Update the current solution and its performance
                current_solution = np.copy(candidate_solution)
                current_performance = candidate_performance

                # Add acceptance mark
                if self.parameters['verbose']:
                    print('A', end='')

            # If the candidate solution is better or equal than the current best solution
            if candidate_performance <= best_performance:

                # Update the best solution and its performance
                best_solution = np.copy(candidate_solution)
                best_performance = candidate_performance

                # Reset the stagnation counter
                stag_counter = 0

                # Save this information
                _save_step(step, {
                    'encoded_solution': best_solution,
                    'performance': best_performance,
                    'details': candidate_details
                }, self.file_label)

                # Add improvement mark
                if self.parameters['verbose']:
                    print('+', end='')
            else:
                # Update the stagnation counter
                stag_counter += 1

            historical_current.append(current_performance)
            historical_best.append(best_performance)
            # Add ending mark
            if self.parameters['verbose']:
                print('')

        # Print the best one
        if self.parameters['verbose']:
            print('\nBEST --> Perf: {}, e-Sol: {}'.format(best_performance, best_solution))

        # Return the best solution found and its details
        return best_solution, best_performance, historical_current, historical_best

    def _solve_dynamic(self, kw_weighting_params, save_steps=True):
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

        :returns: solution (list), performance (float), encoded_solution (list)

        TODO: Update
        """
        sequence_per_repetition = list()
        fitness_per_repetition = list()
        weights_per_repetition = list()

        # Initialise the current performance from the initial guess sequence
        #current_performance = initial_guess_info['performance']

        for rep in range(self.parameters['num_replicas']):
            # Call the metaheuristic
            # mh = None
            mh = Metaheuristic(self.problem, num_agents=self.parameters['num_agents'],
                               num_iterations=self.num_iterations)

            # %% INITIALISER PART
            mh.apply_initialiser()

            # Extract the population and fitness vealues, and their best values
            current_fitness = np.copy(mh.pop.global_best_fitness)
            current_position = np.copy(mh.pop.rescale_back(mh.pop.global_best_position))

            # Heuristic sets
            # tabu_set = list()
            # active_set = list()
            self.current_space = np.arange(self.num_operators)

            # Initialise some additional variables
            candidate_enc_so = list()  # This is a list of up to 1-length
            current_sequence = [-1]

            best_fitness = [current_fitness]
            best_position = [current_position]
            fitness_data = [np.copy(mh.pop.fitness)]
            positions_data = [np.copy(mh.pop.get_positions())]

            step = 0
            stag_counter = 0
            # temperature = self.parameters['max_temperature']

            # # Print the first status update, step = 0
            # if self.parameters['verbose']:
            #     print('{} :: Step: {:4d}, Action: {:12s}, Temp: {:.2e}, Card: {:3d}, Perf: {:.2e} [Initial]'.format(
            #         self.file_label, step, 'None', temperature, len(current_solution), current_performance))

            # We assume that with only one operator, it never reaches the solution. So, we check finalisation ending itr

            # FINALISATOR: Finalise due to other concepts
            while not self._check_finalisation(step, stag_counter):
                # Update the current set
                # self.current_space = np.union1d(np.setdiff1d(self.current_space, tabu_set), active_set).astype(int)
                if self.parameters['trial_overflow'] and self.__stagnation_check(stag_counter):
                    possible_transitions = np.ones(self.num_operators) / self.num_operators
                else:
                    if not ((self.parameters['allow_weight_matrix'] is None) or (self.transition_matrix is None)):
                        # possible_transitions = self.transition_matrix[current_sequence[-1] + 1, 1:]
                        if step < self.transition_matrix.shape[0]:
                            possible_transitions = self.transition_matrix[step]
                            transitions_sum = possible_transitions.sum()
                            if transitions_sum > 0.0:
                                possible_transitions = np.nan_to_num(possible_transitions / transitions_sum)
                            else:
                                possible_transitions = np.ones(self.num_operators) / self.num_operators
                        else:
                            possible_transitions = np.ones(self.num_operators) / self.num_operators

                    else:
                        possible_transitions = self.weights

                # Pick randomly a simple heuristic
                candidate_enc_so = self._obtain_candidate_solution(sol=1, operators_weights=possible_transitions)

                # Prepare before evaluate the last search operator and apply it
                candidate_search_operator = self.get_operators([candidate_enc_so[-1]])
                perturbators, selectors = Operators.process_operators(candidate_search_operator)

                mh.apply_search_operator(perturbators[0], selectors[0])

                # Extract the population and fitness values, and their best values
                current_fitness = np.copy(mh.pop.global_best_fitness)
                current_position = np.copy(mh.pop.rescale_back(mh.pop.global_best_position))

                # Print update
                if self.parameters['verbose']:
                    print(
                        '{} :: Rep: {:3d}, Step: {:3d}, Trial: {:3d}, SO: {:30s}, currPerf: {:.2e}, candPerf: {:.2e}, '
                        'csl: {:3d}'.format(
                            self.file_label, rep, step, stag_counter,
                            candidate_search_operator[0][0] + ' & ' + candidate_search_operator[0][2][:4],
                            best_fitness[-1], current_fitness, len(self.current_space)), end=' ')

                # If the candidate solution is better or equal than the current best solution
                if current_fitness <= best_fitness[-1]:

                    # Reward the selected search operator from the heuristic set
                    # active_set.append(candidate_enc_so)

                    # Update the current sequence and its characteristics
                    current_sequence.append(candidate_enc_so[-1])

                    best_fitness.append(current_fitness)
                    best_position.append(current_position)
                    fitness_data.append(np.copy(mh.pop.fitness))
                    positions_data.append(np.copy(mh.pop.get_positions()))

                    # Update counters
                    step += 1
                    stag_counter = 0
                    self._trial_overflow = False
                    # candidate_enc_so = list()

                    # Add improvement mark
                    if self.parameters['verbose']:
                        print('+', end='')

                else:  # Then try another search operator

                    # Revert the modification to the population in the mh object
                    mh.pop.revert_positions()

                    # Update stagnation
                    stag_counter += 1

                # Add ending mark
                if self.parameters['verbose']:
                    print('')

            # Print the best one
            if self.parameters['verbose']:
                print('\nBest fitness: {},\nBest position: {}'.format(current_fitness, current_position))

            #  Update the repetition register
            sequence_per_repetition.append(np.double(current_sequence).astype(int).tolist())
            fitness_per_repetition.append(np.double(best_fitness).tolist())

            # Update the weights for learning purposes
            weight_matrix = self._update_weights(sequence_per_repetition, fitness_per_repetition,
                                                 **kw_weighting_params)
            weights_per_repetition.append(self.weights)
            # print('w = ({})'.format(self.weights))

            # Save this historical register
            if save_steps:
                _save_step(rep + 1,  # datetime.now().strftime('%Hh%Mm%Ss'),
                           dict(encoded_solution=np.array(current_sequence),
                                best_fitness=np.double(best_fitness),
                                best_positions=np.double(best_position),
                                details=dict(
                                    fitness_per_rep=fitness_per_repetition,
                                    sequence_per_rep=sequence_per_repetition,
                                    weight_matrix=weight_matrix
                                )),
                           self.file_label)

        # Return the best solution found and its details
        return fitness_per_repetition, sequence_per_repetition, weights_per_repetition, weight_matrix

    def _solve_neural_network(self, kw_nn_params, save_steps=True):
        """
        This method perform the implementation using a previously trained nn

        @requires:

        @return:
        TODO: Complete it
        """
        sequence_per_repetition = list()
        fitness_per_repetition = list()
        weights_per_repetition = list()

        # Neural network model that predicts operators
        model, encoder = self._get_neural_network_model(kw_nn_params['model_params'])

        need_ragged_tensor = True if kw_nn_params['model_params']['model_architecture'] == 'LSTM_Ragged' else False

        for rep in range(kw_nn_params['num_replicas']):
            # Metaheuristic
            mh = Metaheuristic(self.problem, num_agents=self.parameters['num_agents'], num_iterations=self.num_iterations)

            # Initialiser
            mh.apply_initialiser()

            # Extract the population and fitness values, and their best values
            current_fitness = np.copy(mh.pop.global_best_fitness)
            current_position = np.copy(mh.pop.rescale_back(mh.pop.global_best_position))

            # Heuristic sets
            self.current_space = np.arange(self.num_operators)

            # Initialise additional variables
            candidate_enc_so = list()
            current_sequence = [-1]

            best_fitness = [current_fitness]
            best_position = [current_position]

            step = 0
            stag_counter = 0
            exclude_indices = []

            # Finalisator
            while not self._check_finalisation(step, stag_counter):
                # Use the trained model to predict operators weights
                operators_weights = self._predict_operator_weights(model, 
                                                                   encoder,
                                                                   current_sequence,
                                                                   exclude_indices,
                                                                   need_ragged_tensor)

                # Select a simple heuristic and apply it
                candidate_enc_so = self._obtain_candidate_solution(sol=1, operators_weights=operators_weights)
                candidate_search_operator = self.get_operators([candidate_enc_so[-1]])
                perturbators, selectors = Operators.process_operators(candidate_search_operator)

                mh.apply_search_operator(perturbators[0], selectors[0])

                # Extract population and fitness values
                current_fitness = np.copy(mh.pop.global_best_fitness)
                current_position = np.copy(mh.pop.rescale_back(mh.pop.global_best_position))

                # Print update
                if self.parameters['verbose']:
                    print(
                        '{} :: Neural Network, Rep: {:3d}, Step: {:3d}, Trial: {:3d}, SO: {:30s}, currPerf: {:.2e}, candPerf: {:.2e}, '
                        'csl: {:3d}'.format(
                            self.file_label, rep, step, stag_counter,
                            candidate_search_operator[0][0] + ' & ' + candidate_search_operator[0][2][:4],
                            best_fitness[-1], current_fitness, len(self.current_space)), end=' ')

                # If the candidate solution is better or equal than the current best solution
                if current_fitness < best_fitness[-1]:
                    # Update the current sequence and its characteristics
                    current_sequence.append(candidate_enc_so[-1])

                    best_fitness.append(current_fitness)
                    best_position.append(current_position)

                    # Update counters
                    step += 1
                    stag_counter = 0
                    # Reset tabu list
                    exclude_indices = []

                    # Add improvement mark
                    if self.parameters['verbose']:
                        print('+', end='')

                else:  # Then try another search operator
                    # Revert the modification to the population in the mh object
                    mh.pop.revert_positions()

                    # Update stagnation
                    stag_counter += 1
                    if stag_counter % kw_nn_params['delete_idx'] == 0:
                        # Include last search operator's index to the tabu list
                        exclude_indices.append(candidate_enc_so[-1])

                # Add ending mark
                if self.parameters['verbose']:
                    print('')

            # Print the best one
            if self.parameters['verbose']:
                print('\nBest fitness: {},\nBest position: {}'.format(current_fitness, current_position))

            # Update the repetition register
            sequence_per_repetition.append(np.double(current_sequence).astype(int).tolist())
            fitness_per_repetition.append(np.double(best_fitness).tolist())

            # Update the weights for learning purposes
            weight_matrix = self._update_weights(sequence_per_repetition, learning_portion=0)
            weights_per_repetition.append(self.weights)

            # Save this historical register
            if save_steps:
                _save_step(rep,
                           dict(encoded_solution=np.array(current_sequence),
                                best_fitness=np.double(best_fitness),
                                best_positions=np.double(best_position),
                                details=dict(
                                    fitness_per_rep=fitness_per_repetition,
                                    sequence_per_rep=sequence_per_repetition,
                                    weight_matrix=weight_matrix
                                )),
                           self.file_label)

        return fitness_per_repetition, sequence_per_repetition, weights_per_repetition, weight_matrix
    

    def _get_neural_network_model(self, kw_model_params):
        """
        Generates a model that learns patterns to predict which search operator applies.

        :param dict kw_model_params:
            Parameters to load or train a neural network model to predict search operators
            
                :param bool load_model : Load a pre-trained model if true
                    
                :param bool save_model : Save trained model if true

                :param str model_id : Identificator for the model

                :param str encoder : Encoder used to encode each sequence

                :param dict autoencoder_architecture : Param only required if encoder is 'autoencoder'
                                    Requires 'latent_dim' variable, the number of epochs and the 
                                    architecture for the encoder and decoder

                :param dict kw_weighting_params : Params for _solve_dynamic

                :param bool include_fitness : Generates sample_weights if true

                :param str fitness_to_weight : Param only required if include_fitness is true
                                    Name of function that would be used to transform fitness to
                                    weight for each fitness sample
                
                :param str model_architeture : Architecture that would be used for the model (MLP or LSTM)

                :param dict model_architecture_layers : Array with size and activaction function for each
                                    layer that would be included in the model, besides the input 
                                    and output layer
                
                :param int epochs: Number of epochs to train the neural network model

        :return: Return a trained model and an encoder function that transform a dynamic sequence of
            search operators to the input of the trained model
        """
        # TODO: Update above description
        core_config = tf.compat.v1.ConfigProto()
        core_config.gpu_options.allow_growth = True
        session = tf.compat.v1.Session(config=core_config)
        keras.backend.set_session(session)


        architecture_name = kw_model_params['model_architecture']
        encoder_name = kw_model_params['encoder']

        model_directory = './ml_models'

        if isinstance(kw_model_params['sample_sequences_params']['classification'], str):
            model_label = kw_model_params['sample_sequences_params']['classification']
            if model_label == '':
                model_label = self.file_label
        else:
            now = datetime.now()
            model_label = f'Exp-{now.strftime("%m_%d_%Y")}'
        
        model_path = '/'.join([model_directory, f'{model_label}.h5'])
        autoencoder_path = '/'.join([model_directory, f'{model_label}_autoencoder.h5'])

        # If there is a model, use it
        if kw_model_params['load_model'] and _check_path(model_path):
            # Check if needs an autoencoder and if it exists or not
            if encoder_name != 'autoencoder' or _check_path(autoencoder_path):
                if encoder_name == 'autoencoder':
                    self.autoencoder = tf.keras.models.load_model(autoencoder_path)
                model = tf.keras.models.load_model(model_path)
                encoder = self._get_encoder(encoder_name)
                return model, encoder

        # Data generation
        seqfitness, seqrep = self._get_sample_sequences(kw_model_params['sample_sequences_params']) 
        X, y, sample_fitness = self.__process_sample_sequences(seqfitness, seqrep)
        
        if encoder_name == 'autoencoder':
            # Prepare parameters for autoencoder
            kw_model_params['autoencoder_architecture']['input_shape'] = self.parameters['num_steps']

            # Prepare training data for autoencoder
            X_train = [self.__padded_identity_encoder(sequence) / self.num_operators for sequence in X]

            # Get encoder 
            autoencoder = _create_autoencoder(X_train, kw_model_params['autoencoder_architecture'])
            self.autoencoder = autoencoder.encoder
        
        # Encode training data
        encoder = self._get_encoder(encoder_name)
        X = [encoder(sequence) for sequence in X]
        input_size = len(X[0])

        # Tensor conversion
        if architecture_name in ['LSTM_Ragged']:
            X = tf.ragged.constant(X)
        else:
            X = tf.constant(X)
        y = tf.constant(y)

        # Include fitness for training
        if kw_model_params['include_fitness']:
            sample_weight = tf.constant(self._obtain_sample_weight_from_fitness(sample_fitness, kw_model_params['fitness_to_weight']))
        else:
            sample_weight = None


        # TensorFlow Artificial Neural Network Model

        # Create model
        model = tf.keras.Sequential()
        hidden_layers = kw_model_params['model_architecture_layers']
        if architecture_name in ['MLP', 'Multi-Layer Perceptron', 'default']:
            # Multi-Layer Perceptron model

            # Input layer
            model.add(tf.keras.Input(shape=input_size))
            
            # Hidden layers
            for layer_size, layer_activation in hidden_layers:
                model.add(tf.keras.layers.Dense(units=layer_size,
                                                activation=layer_activation))
        
        elif architecture_name in ['RNN', 'LSTM', 'Long Short-Term Memory', 'LSTM_Ragged']:
            # Long Short-Term Memory model

            # Input layer
            if architecture_name in ['LSTM_Ragged']:
                # Input with variable length, supported using ragged tensors
                max_length_sequence = X.bounding_shape()[-1].numpy()
                model.add(tf.keras.Input(shape=(max_length_sequence,)))

                # Embedding layer
                first_layer_size, _ = hidden_layers[0] if len(hidden_layers) > 0 else (self.num_operators, None)
                model.add(tf.keras.layers.Embedding(self.num_operators + 1, first_layer_size))
            else:
                # Input with fixed length, sequences padded with dummy values
                model.add(tf.keras.Input(shape=(input_size, 1)))

            # Hidden layers
            num_layers = len(hidden_layers)
            for idx, (layer_size, layer_activation) in enumerate(hidden_layers):
                model.add(tf.keras.layers.LSTM(units=layer_size, 
                                               activation=layer_activation, 
                                               return_sequences=True if idx + 1 < num_layers else False))

        # Output layer
        model.add(tf.keras.layers.Dense(self.num_operators, activation='softmax'))

        # Compile model
        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=['accuracy'])
        
        # Train model
        model.fit(X, y, epochs=kw_model_params['epochs'], sample_weight=sample_weight, verbose=self.parameters['verbose'])

        # Save model
        if kw_model_params['save_model']:
            if not _check_path(model_directory):
                _create_path(model_directory)
            model.save(model_path)
            if encoder_name == 'autoencoder':
                autoencoder.encoder.save(autoencoder_path)

        session.close()
        keras.backend.clear_session()

        return model, encoder

    def __process_sample_sequences(self, seqfitness, seqrep):        
        "Process sequences to generate data for training"
        
        X, y, sample_fitness = [], [], []
        for fitness, sequence in zip(seqfitness, seqrep):
            if len(sequence) > 0 and sequence[0] == -1:
                sequence.pop(0)
                fitness.pop(0)
            while len(sequence) > 0:
                # Per each prefix, predict the next operator
                y.append(self.__one_hot_encoding_sequence([sequence.pop()]))
                X.append(sequence.copy())
                sample_fitness.append(fitness.pop())

        return X, y, sample_fitness

    def _predict_operator_weights(self, model, encoder, sequence, exclude_indices, ragged=False):
        """
        Given the state of a dynamic sequence, generate an array of probabilities to choose the next search operator

        :param tf.keras.model model:
            Trained model to predict a weighted array

        :param function encoder:
            Function that encode a sequence to a valid input for the trained model

        :param list sequence:
            List of search operator's indices that represents the dynamic sequence

        :param list exclude_indices:
            Tabu list of indices that are excluded
        
        :param bool ragged:
            Specify if it is necessary to use Ragged Tensors or not
        
        :return: List of probabilities to choose the next search operator
        """
        # Use model to predict weights
        encoded_sequence = []
        if ragged:
            encoded_sequence = tf.ragged.constant([encoder(sequence)])
        else:
            encoded_sequence = tf.constant([encoder(sequence)])

        output_weights = model.predict(encoded_sequence)[0]

        # Exclude tabu indices
        for idx in exclude_indices:
            output_weights[idx] = 0

        # Set probability weights 
        if sum(output_weights) > 0:
            operators_weights = output_weights / sum(output_weights)
        else:
            operators_weights = np.ones(self.num_operators) / self.num_operators

        return operators_weights

    def evaluate_candidate_solution(self, encoded_sequence):
        """
        Evaluate the current sequence as a hyper/meta-heuristic. This process is repeated ``parameters['num_replicas']``
        times and, then, the performance is determined. In the end, the method returns the performance value and the
        details for all the runs. These details are ``historical_data``, ``fitness_data``, ``position_data``, and
        ``fitness_stats``. The elements from the ``encoded_sequence`` must be in the range of the ``num_operators``.

        :param list encoded_sequence:
            Sequence of search operators. These must be in the tuple form (decoded version). Check the ``metaheuristic``
            module for further information.
        :return: float, dict
        """
        # Decode the sequence corresponding to the hyper/meta-heuristic
        search_operators = self.get_operators(encoded_sequence)

        # Initialise the historical registers
        historical_data = list()
        fitness_data = list()
        position_data = list()

        # Run the metaheuristic several times
        for rep in range(self.parameters['num_replicas']):
            # Call the metaheuristic
            mh = Metaheuristic(self.problem, search_operators, self.parameters['num_agents'], self.num_iterations)

            # Run this metaheuristic
            mh.run()

            # Store the historical values from this run
            historical_data.append(mh.historical)

            # Read and store the solution obtained
            _temporal_position, _temporal_fitness = mh.get_solution()
            fitness_data.append(_temporal_fitness)
            position_data.append(_temporal_position)
            # print('-- MH: {}, fitness={}'.format(rep + 1, _temporal_fitness))

        # Determine a performance metric once finish the repetitions
        fitness_stats = self.get_statistics(fitness_data)

        # Return the performance value and the corresponding details
        return self.get_performance(fitness_stats), dict(
            historical=historical_data, fitness=fitness_data, positions=position_data, statistics=fitness_stats)

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
            operator_performance, operator_details = self.evaluate_candidate_solution(operator)

            # Save information
            _save_step(operator_id, {
                'encoded_solution': operator_id,
                'performance': operator_performance,
                'statistics': operator_details['statistics']
            }, self.file_label)

            # Print update
            print('{} :: Operator {} of {}, Perf: {}'.format(
                self.file_label, operator_id + 1, self.num_operators, operator_performance))

    def basic_metaheuristics(self):
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
            operator_performance, operator_details = self.evaluate_candidate_solution(operator)

            # Save information
            _save_step(operator_id, {
                'encoded_solution': operator_id,
                'performance': operator_performance,
                'statistics': operator_details['statistics']
                # 'fitness': operator_details['fitness'],  # to-delete
                # 'historical': operator_details['historical']  # to-delete
            }, self.file_label)

            # Print update
            print('{} :: BasicMH {} of {}, Perf: {}'.format(
                self.file_label, operator_id + 1, self.num_operators, operator_performance))

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
        return statistics['Med'] + statistics['IQR']  # Option 3
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
                    MAD=st.median_abs_deviation(raw_data))

    # Script for interpolating matrices
    @staticmethod
    def __interpolate_matrices(matrix1, matrix2, blend_factor=0.5):
        return matrix1 * (1 - blend_factor) + matrix2 * blend_factor

    # Script for determining the dimensionless point to interpolate
    @staticmethod
    def __get_blend_factor(extrema, point):
        return (point - extrema[0]) / (extrema[1] - extrema[0])

    def _get_sample_sequences(self, kw_sequences_params):
        """
        Retrieve or generate sequences to use them as data train

        :param dict kw_sequences_params: Store the following values,
        
            :param bool retrieve_sequences: True if would retrieve stored sequences
            
            :param bool generate_sequences: True if would generate sequences for the given problem
            
            :param str classification: Specify which kind of sequences would be retrived
            
            :param dict kw_weighting_params: Specify the parameters to run the dynamic solver over the given problem
            
            :param bool store_sequences: True if would store the sample of sequences
            
        :return: A sample of sequences and their fitness for training
        """

        # Initialize variables
        seqfitness_retrieved, seqrep_retrieved = [], []
        seqfitness_generated, seqrep_generated = [], []
        
        # Obtain sequences from previous generations
        if kw_sequences_params['retrieve_sequences']:
            if isinstance(kw_sequences_params['classification'], str) and len(kw_sequences_params['classification']) == 0:
                # Temporal fix to run third experiment
                sections_labels = self.file_label.split('-')
                sequences_folder = '-'.join(sections_labels[:2]+['short_nn_mlp_learning.json'])
                seqfitness_retrieved, seqrep_retrieved = _get_stored_sample_sequences(sequences_folder)
            else:
                seqfitness_retrieved, seqrep_retrieved = _get_stored_sample_sequences(kw_sequences_params['classification'])
                
            
        # Generate sequences from dynamic solver
        if kw_sequences_params['generate_sequences'] or len(seqfitness_retrieved) == 0:
            seqfitness_generated, seqrep_generated, _, _ = self._solve_dynamic(kw_sequences_params['kw_weighting_params'], save_steps=False)        
        
        # Join sequences
        seqfitness = seqfitness_retrieved + seqfitness_generated
        seqrep = seqrep_retrieved + seqrep_generated
        
        # Verify that there is sequences for training
        if len(seqfitness) == 0 or len(seqrep) == 0:
            raise HyperheuristicError('There is no sample sequences for training')

        # Store sequences if requested
        if kw_sequences_params['store_sequences']:        
            sequences_to_save = dict()
            for idx, fitness_and_sequence in enumerate(zip(seqfitness, seqrep)):
                sequences_to_save[idx] = fitness_and_sequence
            _save_sequences(self.file_label, sequences_to_save)
    
        return seqfitness, seqrep

    def _get_weight_matrix(self, category, values_dict, file_path="./data_files/translearn_dataset"):

        # Load the datafile
        dataset = pd.read_json(file_path)
        # category = 0, values_dict = {"Dim": 3, "Pop": 31}

        # Check if the values are in the list, then return a list with its neighbours
        limits = dict()
        for key, val in values_dict.items():
            available_values = dataset[key].unique().tolist()
            if val not in available_values:
                # TODO: improve for extrapolations
                available_values.append(val)
                available_values.sort()
                index = available_values.index(val)
                limits_values = available_values[index - 1:index + 2:2]
            else:
                limits_values = [val]

            limits[key] = limits_values
            print(limits)

        comb = [*product(*limits.values())]
        num_comb = len(comb)
        # Read matrices
        matrices = list()
        for dim_val, pop_val in comb:
            matrices.append(np.array(*dataset[
                (dataset['Dim'] == dim_val) &
                (dataset['Pop'] == pop_val) &
                (dataset['Cat'] == category)]['weights'].tolist()))
            print(dim_val, pop_val)

        if num_comb == 1:
            # Just pick up the corresponding matrix
            out_matrix = matrices[0]
        elif num_comb == 2:
            # Find the intermediate factor to interpolate
            if len(limits['Pop']) == 2:
                factor = self.__get_blend_factor(limits['Pop'], values_dict['Pop'])
            elif len(limits['Dim']) == 2:
                factor = self.__get_blend_factor(limits['Dim'], values_dict['Dim'])
            else:
                factor = None
                raise HyperheuristicError("Invalid case for determining factor")

            # Perform one interpolation
            out_matrix = self.__interpolate_matrices(matrices[0], matrices[1], factor)

        elif num_comb == 4:
            # Find the corresponding factors
            pop_factor = self.__get_blend_factor(limits['Pop'], values_dict['Pop'])
            dim_factor = self.__get_blend_factor(limits['Dim'], values_dict['Dim'])

            # Perform two pre-interpolations with pop values
            prematrix_pop1 = self.__interpolate_matrices(matrices[0], matrices[1], pop_factor)
            prematrix_pop2 = self.__interpolate_matrices(matrices[2], matrices[3], pop_factor)

            # Perform the last interpolation with dim values
            out_matrix = self.__interpolate_matrices(prematrix_pop1, prematrix_pop2, dim_factor)
        else:
            out_matrix = None
            raise HyperheuristicError("Invalid case for the number of combinations")

        # Adjusting the matrix to avoid values out of range
        total_count_per_step = out_matrix.sum(1)
        out_weights = (out_matrix.T / total_count_per_step).T

        return out_weights

    def _obtain_sample_weight_from_fitness(self, sample_fitness, fitness_to_weight):
        """
        Using decreasing functions to give more priority to samples with less fitness

        :param list sample_fitness:
            The fitness associated value for each sample
        
        :param str fitness_to_weight:
            Specify which function use to convert fitness to weight
        
        :return: An array that associates a weight to each sample
        """
        a = min(sample_fitness)
        b = max(sample_fitness)
        if fitness_to_weight == 'linear_reciprocal':
            # f: [a, b] -> (0, 1]
            weight_conversion = lambda fitness: a / fitness
        elif fitness_to_weight == 'linear_reciprocal_translated':
            # f: [a, b] -> [1, b-a+1]
            weight_conversion = lambda fitness: a * b / fitness - a + 1
        elif fitness_to_weight == 'linear_percentage':
            # f: [a, b] -> [1, 100]
            weight_conversion = lambda fitness: 100 * (b - fitness) / (b - a) + 1
        elif fitness_to_weight == 'rank':
            # f: [fitness] -> [0, n-1]
            indices = list(range(len(sample_fitness)))
            indices.sort(key = lambda idx: -sample_fitness[idx])
            return indices
        else:
            # Default linear conversion
            # f: [a, b] -> [a, b]
            weight_conversion = lambda fitness: a + b - fitness
        
        return [weight_conversion(fitness) for fitness in sample_fitness]

    def _update_weights(self, sequences=None, fitness_values=None, include_fitness=False, learning_portion=0.37):
        # *** uncomment when necessary
        # if not (isinstance(sequences, list) and isinstance(sequences[0], list)):
        #     sequences = [sequences]
        #
        # if not (isinstance(fitness_values, list) and isinstance(fitness_values[0], list)):
        #     fitness_values = [fitness_values]

        if (self.weights is None) or (len(sequences) < int(self.parameters['num_replicas'] * learning_portion
                                                           ) if sequences is not None else False):
            # create the weights array using a uniform distribution
            self.weights = np.ones(self.num_operators) / self.num_operators
            self.__total_count_weights = self.num_operators
        else:
            # update the array [q.count(x) for x in range(min(q), 1 + max(q))]
            pre_weights = list()
            weimat = np.zeros([self.num_operators + 1] * 2)

            # Get the matrix from sequences of num_operators -by- num_steps. Empties are filled with -2
            max_length = max([len(seq) for seq in sequences])
            mat_seq = np.array([np.array([*seq, *[-2] * (max_length - len(seq))]) for seq in sequences],
                               dtype=object).T

            all_operators_including_empty = [-2.5, *np.arange(-2, self.num_operators) + 0.5]
            current_hist = list()
            for ii_step in range(max_length):
                # Disregard the -2 and -1 operators (empty and initialiser)
                densities, _ = np.histogram(mat_seq[ii_step].tolist(), bins=all_operators_including_empty)
                temp_hist = densities[2:]
                if np.sum(temp_hist) > 0.0:
                    current_hist.append(np.ndarray.tolist(temp_hist / np.sum(temp_hist)))
                else:
                    current_hist.append(np.ndarray.tolist(np.ones(self.num_operators) / self.num_operators))

            weimat = np.array(current_hist)

            weights_to_update = np.array(pre_weights).sum(axis=0) + self.weights * self.__total_count_weights
            self.__total_count_weights = weights_to_update.sum()
            self.weights = weights_to_update / self.__total_count_weights
            self.transition_matrix = weimat

            return weimat

    def __fill_sequence(self, sequence):
        "Fill a sequence with a dummy value until a fixed length"
        return np.array(np.pad(sequence, 
                               (self.parameters['num_steps'] - len(sequence), 0), 
                               constant_values=self.num_operators).astype(int))

    def __identity_encoder(self, sequence):
        "Clean a sequence to encode it"
        sequence_copy = sequence.copy()
        if len(sequence_copy) > 0 and sequence_copy[0] == -1:
            sequence_copy.pop(0)
        return sequence_copy
    
    def __padded_identity_encoder(self, sequence):
        "Clean and fill a sequence to encode it"
        return self.__fill_sequence(self.__identity_encoder(sequence))

    def __one_hot_encoding_sequence(self, sequence):
        "One-Hot encoding a sequence of search operator's indices"
        flatten = lambda arr: [int(item) for list_arr in arr for item in list_arr]
        return flatten(tf.one_hot(indices=sequence, depth=self.num_operators).numpy())
    
    def __one_hot_encoder(self, sequence):
        "Fill sequence and apply one-hot encoding to encode a sequence"
        return self.__one_hot_encoding_sequence(self.__padded_identity_encoder(sequence))

    def __autoencoder_encoder(self, sequence):
        "Use encoder from autoencoder model to encode a sequence"
        if self.autoencoder is None:
            raise HyperheuristicError('Autoencoder does not exists')
        sequence = self.__padded_identity_encoder(sequence) / self.num_operators
        return self.autoencoder(tf.constant([sequence])).numpy()[0]

    def __lstm_identity_encoder(self, sequence):
        "Reshape sequence to use it in the LSTM model"
        padded_sequence = self.__padded_identity_encoder(sequence)
        return [[x] for x in padded_sequence]

    def __lstm_one_hot_encoder(self, sequence):
        "Reshape one-hot encoded sequence to use it in the LSTM model"
        one_hot_sequence = self.__one_hot_encoder(sequence)
        return [[x] for x in one_hot_sequence]

    def __lstm_ragged_identity_encoder(self, sequence):
        "Processing sequence to use it in a LSTM model based on ragged tensors"
        cleaned_sequence = self.__identity_encoder(sequence)
        # LSTM Ragged tensor needs a non-empty tensor to predict an operator
        if len(cleaned_sequence) == 0:
            cleaned_sequence.append(self.num_operators)
        return cleaned_sequence

    def __lstm_ragged_one_hot_encoder(self, sequence):
        "One-Hot encoding processed sequence for LSTM model based on ragged tensor"
        return self.__one_hot_encoding_sequence(self.__lstm_ragged_identity_encoder(sequence))

    def _get_encoder(self, encoder_name):
        if encoder_name == 'identity':
            return self.__identity_encoder
        elif encoder_name == 'one_hot_encoder':
            return self.__one_hot_encoder
        elif encoder_name == 'autoencoder':
            return self.__autoencoder_encoder
        elif encoder_name in ['LSTM', 'LSTM_identity']:
            return self.__lstm_identity_encoder
        elif encoder_name in ['LSTM_one_hot']:
            return self.__lstm_one_hot_encoder
        elif encoder_name in ['LSTM_Ragged', 'LSTM_Ragged_identity', 'Ragged']:
            return self.__lstm_ragged_identity_encoder
        elif encoder_name in ['LSTM_Ragged_one_hot']:
            return self.__lstm_ragged_one_hot_encoder
        else:
            # Default encoder
            return self.__padded_identity_encoder

# %% ADDITIONAL TOOLS

def _save_step(step_number, variable_to_save, prefix=''):
    """
    This method saves all the information corresponding to specific step.

    :param int|str step_number:
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
    if prefix != '':
        folder_name = 'data_files/raw/' + prefix
    else:
        folder_name = 'data_files/raw/' + 'Exp-' + now.strftime('%m_%d_%Y')

    # Check if this path exists
    if not _check_path(folder_name):
        _create_path(folder_name)

    # Create a new file for this step
    with open(folder_name + f'/{str(step_number)}-' + now.strftime('%m_%d_%Y_%H_%M_%S') + '.json', 'w') as json_file:
        json.dump(variable_to_save, json_file, cls=jt.NumpyEncoder)

def _get_problems_in_classification(classification):
    "Return a list of problems that satifies such classification"
    #TODO: Consider move this function to benchmark_func.py
    #TODO: Support differentiation between problem and dimension
    import benchmark_func as bf

    process_list = []
    problems_list = []

    if isinstance(classification, str):
        process_list = [classification]
    elif isinstance(classification, list):
        process_list = classification
    elif isinstance(classification, dict):
        # Dict with keys 'func_names', 'dim', 'features'
        # Not supported yet
        return []
        
    if len(process_list) > 0:
        for category in process_list:
            if '-' in category:
                # {func_name}-{variable_num}D
                func_name = category.split('-')
                problems_list.append(func_name)
            elif category in bf.__all__:
                # func_name
                problems_list.append(category)
            else:
                # Feature list
                
                return []
                    
    return problems_list
    
def _get_stored_sample_sequences(classification):
    "Retrieve stored sequences"
    # TODO: Fix function : Currently broken until fix classification
    folder_name = 'data_files/sequences/'
    if not _check_path(folder_name):
        return [], []

    problems_in_classification = _get_problems_in_classification(classification)
    file_names = _get_problems_in_classification(classification)

    sequences_file = folder_name + f'{classification}.json'
    if not _check_path(folder_name) or not _check_path(sequences_file):
        return [], []

    with open(sequences_file, 'r') as json_file:
        sequences_stored = json.load(json_file)
        seqfitness, seqrep = [], []
        for key in sequences_stored:
            fitness, sequence = sequences_stored[key]
            seqfitness.append(fitness)
            seqrep.append(sequence)
    return seqfitness, seqrep

def _save_sequences(file_name, sequences_to_save):
    """
    Save encoded sequences along its fitness to train ML models. 

    :param str file_name
    :param dict sequences_to_save

    :return:
    """
    # Define the folder name
    folder_name = 'data_files/sequences/'
    
    # Check if this path exists
    if not _check_path(folder_name):
        _create_path(folder_name)
    
    # Overwrite or create file to store the sequences along its respective fitness
    with open(folder_name + f'{file_name}.json', 'w') as json_file:
        json.dump(sequences_to_save, json_file, cls=jt.NumpyEncoder)
        
class HyperheuristicError(Exception):
    """
    Simple HyperheuristicError to manage exceptions.
    """
    pass


if __name__ == '__main__':
    # import hyperheuristic as hh
    import benchmark_func as bf
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.stats as st
    from pprint import pprint
    import tikzplotlib as ptx
    from sklearn.preprocessing import normalize
    import seaborn as sns

    plt.rcParams.update({'font.size': 18,
                         "text.usetex": True,
                         "font.family": "serif"})

    problem = bf.Sphere(50)
    # problem = bf.Stochastic(50)
    # problem = bf.CosineMixture(50)
    # problem = bf.Whitley(50)
    # problem = bf.Schwefel220(50)
    # problem = bf.Sargan(45)

    problem = bf.choose_problem('<random>', np.random.randint(2, 50))
    # problem.set_search_range(-10, 10)

    # Reproducible for test purposes
    tf.random.set_seed(1)
    np.random.seed(1)

    file_label = "{}-{}D".format(problem.func_name, problem.variable_num)

    q = Hyperheuristic(problem=problem.get_formatted_problem(),
                       heuristic_space='medium_collection.txt',  # 'default.txt',  #short_collection  automatic medium_collection
                       file_label=file_label)
    q.parameters['num_agents'] = 30
    q.parameters['num_steps'] = 100
    q.parameters['stagnation_percentage'] = 0.6
    q.parameters['num_replicas'] = 30
    sampling_portion = 0.37  # 0.37

    # fitprep, seqrep, weights, weimatrix = q.solve('dynamic', {
    #     'include_fitness': False,
    #     'learning_portion': sampling_portion
    # })
    # q.parameters['allow_weight_matrix'] = True
    # q.parameters['trial_overflow'] = True

    kw_neural_network_params = dict({
        'num_replicas': 10,
        'delete_idx': 5,
        'model_params': {
            'load_model': False,
            'save_model': True,
            'sample_sequences_params': {
                'classification': '',
                'retrieve_sequences': False,
                'generate_sequences': True,
                'store_sequences': False,
                'kw_weighting_params': {
                    'include_fitness': False,
                    'learning_portion': 1.0 #sampling_portion
                }
            },
            'encoder': 'Default', 
            'include_fitness': True,
            'fitness_to_weight': 'rank',
            'model_architecture': 'MLP',
            'model_architecture_layers': [
                [50, 'relu'],
                [50, 'relu'],
                [50, 'relu'],
                [50, 'relu']
            ],
            'epochs': 100
        }
    })

    """
    self_train = 3
    model_label = 'Sphere-50D-self_trained_model'
    for _ in range(self_train):
        
        fitprep_nn, seqrep_nn, _, _ = q.solve('neural_network', kw_neural_network_params)

        fitprep_prev, seqrep_prev = q._get_stored_sample_sequences(model_label)

        sequences_to_save = dict()
        for idx, fitness_and_sequence in enumerate(zip(fitprep_nn + fitprep_prev, seqrep_nn + seqrep_prev)):
            sequences_to_save[idx] = fitness_and_sequence
        _save_sequences(model_label, sequences_to_save)
        
        kw_neural_network_params['model_params']['sample_sequences_params']['retrieve_sequences'] = True
        kw_neural_network_params['model_params']['sample_sequences_params']['generate_sequences'] = False
    """

    fitprep_nn, seqrep_nn, weights, weimatrix = q.solve('neural_network', kw_neural_network_params)

    q.parameters['num_replicas'] = 50
    # sampling_portion = 0.37  # 0.37
    fitprep_dyn, seqrep_dyn, _, _ = q.solve('dynamic', {
        'include_fitness': False,
        'learning_portion': sampling_portion
    })

    fitprep = fitprep_dyn.copy()
    for seq in fitprep_nn:
        fitprep.append(seq)
    seqrep = seqrep_dyn.copy()
    for seq in seqrep_nn:
        seqrep.append(seq)

    colours = plt.cm.rainbow(np.linspace(0, 1, len(fitprep)))

    # ------- Figure 0
    # fi0 = plt.figure()
    # plt.ion()
    #
    # # Find max length
    # max_length = max([x.__len__() for x in seqrep])
    # mat_seq = np.array([np.array([*x, *[-2] * (max_length - len(x))]) for x in seqrep], dtype=object).T
    #
    # bins = np.arange(-2, 30 + 1)
    # current_hist = list()
    # for step in range(max_length):
    #     dummy_hist = np.histogram(mat_seq[step, :], bins=bins, density=True)[0][2:]
    #     current_hist.append(dummy_hist)
    #
    # sns.heatmap(np.array(current_hist).T, linewidths=.5, cmap='hot_r')
    #
    # plt.xlabel('Step')
    # # plt.yticks(range(30, step=2), range(start=1, stop=31, step=2))
    # plt.ylabel('Operator')
    # plt.ioff()
    # # plt.plot(c, 'o')
    # plt.show()

    folder_name = './figures-to-export/'
    if not _check_path(folder_name):
        _create_path(folder_name)

    # ------- Figure 1
    fi1 = plt.figure(figsize=(8, 3))
    plt.ion()
    for x, c in zip(fitprep, colours):
        plt.plot(x, '-o', color=c)
    plt.xlabel('Step')
    plt.ylabel('Fitness')
    plt.ioff()
    # plt.plot(c, 'o')
    plt.savefig(folder_name + file_label + "_FitnesStep" + ".svg", dpi=333, transparent=True)
    fi1.show()

    # ------- Figure 2
    fi2 = plt.figure(figsize=(6, 6))
    ax = fi2.add_subplot(111, projection='3d')
    plt.ion()
    for x, y, c in zip(fitprep, seqrep, colours):
        ax.plot3D(range(1, 1 + len(x)), y, x, 'o-', color=c)

    plt.xlabel('Step')
    plt.ylabel('Search Operator')
    ax.set_zlabel('Fitness')
    plt.ioff()
    plt.savefig(folder_name + file_label + "_SOStepFitness" + ".svg", dpi=333, transparent=True)
    fi2.show()

    # ------- Figure 3
    new_colours = plt.cm.jet(np.linspace(0, 1, len(fitprep)))

    """
    fi3 = plt.figure(figsize=(6, 6))
    ax = fi3.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=30)
    plt.ion()
    
    for w, i, c in zip(weights, range(1, len(fitprep) + 1), new_colours):
        ax.plot3D([i] * len(w), range(1, len(w) + 1), w, '-', color=c)
    plt.xlabel('Repetition')
    plt.ylabel('Search Operator')
    ax.set_zlabel('Weight')
    
    plt.ioff()
    fi3.show()
    """

    # ------- Figure 4
    if weimatrix is not None:
        # plt.figure()
        fi3 = plt.figure(figsize=(8, 3))
        plt.imshow(weimatrix.T, cmap="hot_r")
        plt.xlabel('Step')
        plt.ylabel('Search Operator')
        plt.savefig(folder_name + file_label + "_SOStep" + ".svg", dpi=333, transparent=True)
        fi3.show()

    # ------- Figure 5
    # last_fitness_values = np.array([ff[-1] for ff in fitprep])
    # midpoint = int(q.parameters['num_replicas'] * sampling_portion)

    # plt.figure()
    # plt.figure(figsize=(8, 3))
    # plt.boxplot([last_fitness_values[:midpoint], last_fitness_values[midpoint:], last_fitness_values],
    #            showmeans=True)
    # plt.xticks(range(1, 4), ['Train', 'Test/Refine', 'All'])

    get_last_fitness = lambda fitlist: np.array([ff[-1] for ff in fitlist])
    last_fitness_values = get_last_fitness(fitprep)
    last_fitness_values_nn = get_last_fitness(fitprep_nn)
    last_fitness_values_dyn = get_last_fitness(fitprep_dyn)
    midpoint = int(q.parameters['num_replicas'] * sampling_portion)

    fi4 = plt.figure(figsize=(8, 3))
    plt.boxplot([last_fitness_values_nn, last_fitness_values_dyn[:midpoint], last_fitness_values_dyn[midpoint:],
                 last_fitness_values],
                showmeans=True)
    plt.xticks(range(1, 5), ['Neural network', 'Train', 'Test/Refine', 'All'])

    plt.ylabel('Fitness')
    plt.xlabel('Sample')
    plt.savefig(folder_name + file_label + "FitnessSample" + ".svg", dpi=333, transparent=True)
    plt.show()

    # print('Stats for all fitness values:')
    #pprint(st.describe(last_fitness_values[:midpoint])._asdict())
    pprint(st.describe(last_fitness_values_nn)._asdict())
    pprint(st.describe(last_fitness_values_dyn)._asdict())
    pprint(st.describe(last_fitness_values)._asdict())
    #
    # print('Stats for train fitness values:')
    # pprint(st.describe(last_fitness_values)._asdict())
    # import numpy as np
    #
    # qq, _ = q._obtain_candidate_solution(np.array(range(2)), 'RemoveMany')
