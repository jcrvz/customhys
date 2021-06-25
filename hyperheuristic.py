# -*- coding: utf-8 -*-
"""
This module contains the Hyperheuristic class.

Created on Thu Jan  9 15:36:43 2020

@author: Jorge Mario Cruz-Duarte (jcrvz.github.io), e-mail: jorge.cruz@tec.mx
"""
import numpy as np
import random
import scipy.stats as st
from metaheuristic import Metaheuristic
from datetime import datetime
import operators as Operators
import json
import tools as jt
from os.path import exists as _check_path
from os import makedirs as _create_path
from deprecated import deprecated


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
            self.heuristic_space = heuristic_space
        elif isinstance(heuristic_space, str):
            with open('collections/' + heuristic_space, 'r') as operators_file:
                self.heuristic_space = [eval(line.rstrip('\n')) for line in operators_file]
        else:
            raise HyperheuristicError('Invalid heuristic_space')

        # Assign default values
        if not parameters:
            parameters = dict(cardinality=3,                # Max. numb. of SOs in MHs, lvl:1
                              cardinality_min=1,            # Min. numb. of SOs in MHs, lvl:1
                              num_iterations=100,           # Iterations a MH performs, lvl:1
                              num_agents=30,                # Agents in population,     lvl:1
                              as_mh=False,                  # HH sequence as a MH?,     lvl:2
                              num_replicas=50,              # Replicas per each MH,     lvl:2
                              num_steps=200,                # Trials per HH step,       lvl:2
                              stagnation_percentage=0.37,   # Stagnation percentage,    lvl:2
                              max_temperature=1,            # Initial temperature (SA), lvl:2
                              min_temperature=1e-6,         # Min temperature (SA),     lvl:2
                              cooling_rate=1e-3,            # Cooling rate (SA),        lvl:2
                              temperature_scheme='fast',    # Temperature updating (SA),lvl:2 *new
                              acceptance_scheme='exponential',  # Acceptance mode,      lvl:2 *new
                              repeat_operators=True,        # Allow repeating SOs inSeq,lvl:2
                              verbose=True)                 # Verbose process,          lvl:2
        # Read the problem
        if problem:
            self.problem = problem
        else:
            raise HyperheuristicError('Problem must be provided')

        # Read the heuristic space size and create the active set
        self.num_operators = len(self.heuristic_space)
        self.current_space = np.arange(self.num_operators)

        # Read the weights (if it is entered)
        self.weights_array = weights_array

        # Initialise other parameters
        self.parameters = parameters
        self.file_label = file_label

        self.max_cardinality = None
        self.min_cardinality = None
        self.num_iterations = None
        self.toggle_seq_as_meta(parameters['as_mh'])

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

    def _choose_action(self, current_cardinality, previous_action=None):
        # First read the available actions. Those could be ...
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

    def _obtain_candidate_solution(self, sol=None, action=None):
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
        if sol is None:
            initial_cardinality = self.min_cardinality if self.parameters['as_mh'] else \
                (self.max_cardinality + self.min_cardinality) // 2

            encoded_neighbour = np.random.choice(
                self.current_space if (self.weights_array is None) else self.num_operators ,
                initial_cardinality, replace=self.parameters['repeat_operators'], p=self.weights_array)

        # If sol is an integer, assume that it refers to the cardinality
        elif isinstance(sol, int):
            encoded_neighbour = np.random.choice(
                self.current_space if (self.weights_array is None) else self.num_operators, sol,
                replace=self.parameters['repeat_operators'], p=self.weights_array)

        elif isinstance(sol, (np.ndarray, list)):
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

            elif action == 'LocalShift': # It only works with the full set
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
            return self.parameters['max_temperature'] / (1 + (1-self.parameters['cooling_rate']) * np.log(step_val + 1))

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

    def _check_finalisation(self, step, stag_counter, *args):
        """ TODO: update this information and check if step should be here
        General finalisation approach implemented for the methodology working as a hyper-heuristic. It mainly depends on
        `step` (current iteration number) and `stag_counter` (current stagnation iteration number). There are other
         variables that can be considered such as `temperature`. These additional variables must be args[0] <= 0.0.
        """
        return (step > self.parameters['num_steps']) or (
                stag_counter > (self.parameters['stagnation_percentage'] * self.parameters['num_steps'])) or (
                any([var <= 0.0 for var in args]))

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

    def solve(self, mode='static'):
        if mode == 'dynamic':
            return self._solve_dynamic()
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

    def _solve_dynamic(self):
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
        for rep in range(1, self.parameters['num_replicas'] + 1):
            # Call the metaheuristic
            # mh = None
            mh = Metaheuristic(self.problem, num_agents=self.parameters['num_agents'], num_iterations=self.num_iterations)

            # %% INITIALISER PART
            mh.apply_initialiser()

            # Extract the population and fitness values, and their best values
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
            temperature = self.parameters['max_temperature']

            # # Print the first status update, step = 0
            # if self.parameters['verbose']:
            #     print('{} :: Step: {:4d}, Action: {:12s}, Temp: {:.2e}, Card: {:3d}, Perf: {:.2e} [Initial]'.format(
            #         self.file_label, step, 'None', temperature, len(current_solution), current_performance))

            # We assume that with only one operator, it never reaches the solution. So, we check finalisation ending itr

            # FINALISATOR: Finalise due to other concepts
            while not self._check_finalisation(step, stag_counter):
                # Update the current set
                # self.current_space = np.union1d(np.setdiff1d(self.current_space, tabu_set), active_set).astype(int)

                # Pick randomly a simple heuristic, action = (next_action), initially 'Add'
                if len(candidate_enc_so) == 0:
                    candidate_enc_so = self._obtain_candidate_solution(sol=1)
                else:
                    candidate_enc_so = self._obtain_candidate_solution(sol=candidate_enc_so, action='Restart')

                # Prepare before evaluate the last search operator and apply it
                candidate_search_operator = self.get_operators([candidate_enc_so[-1]])
                perturbators, selectors = Operators.process_operators(candidate_search_operator)

                mh.apply_search_operator(perturbators[0], selectors[0])

                # Extract the population and fitness values, and their best values
                current_fitness = np.copy(mh.pop.global_best_fitness)
                current_position = np.copy(mh.pop.rescale_back(mh.pop.global_best_position))

                # Print update
                if self.parameters['verbose']:
                    print('{} :: Step: {:3d}, Trial: {:3d}, SO: {:30s}, currPerf: {:.2e}, candPerf: {:.2e}, csl: {:3d}'.format(
                        self.file_label, step, stag_counter,
                        candidate_search_operator[0][0] + ' & ' + candidate_search_operator[0][2][:4],
                        best_fitness[-1], current_fitness, len(self.current_space)), end=' ')

                # If the candidate solution is better or equal than the current best solution
                if current_fitness < best_fitness[-1]:

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
                    candidate_enc_so = list()

                    # Add improvement mark
                    if self.parameters['verbose']:
                        print('+', end='')

                else:  # Then try another search operator

                    # Remove the selected search operator from the heuristic set
                    # tabu_set.append(candidate_enc_so)
                    # if len(tabu_set) > 100:
                    #     del tabu_set[0]

                    # Revert the modification to the population in the mh object
                    mh.pop.revert_positions()

                    # Update stagnation
                    stag_counter += 1

                # Add ending mark
                if self.parameters['verbose']:
                    print('')

            # Save this historical register
            _save_step(rep,  # datetime.now().strftime('%Hh%Mm%Ss'),
                       dict(encoded_solution=np.array(current_sequence),
                            best_fitness=np.double(best_fitness),
                            best_positions=np.double(best_position),
                            # details=dict(
                            #     positions=np.double(positions_data),
                            #     fitness=np.double(fitness_data))
                            ),
                       self.file_label)

            # Print the best one
            if self.parameters['verbose']:
                print('\nBest fitness: {},\nBest position: {}'.format(current_fitness, current_position))

            #  Update the repetition register
            sequence_per_repetition.append(np.double(current_sequence).astype(int).tolist())
            fitness_per_repetition.append(np.double(best_fitness).tolist())

            # Update the weights for learning purposes
            self._update_weights(sequence_per_repetition, fitness_per_repetition)

        # Return the best solution found and its details
        return fitness_per_repetition, sequence_per_repetition

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
        for rep in range(1, self.parameters['num_replicas'] + 1):
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
            # print('-- MH: {}, fitness={}'.format(rep, _temporal_fitness))

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

    def _update_weights(self, sequences=None, fitness_values=None):
        if self.weights_array is None:
            # create the weights array using a uniform distribution
            self.weights_array = np.ones(self.num_operators) / self.num_operators
        else:
            # update the array [q.count(x) for x in range(min(q), 1 + max(q))]
            pre_weights = list()

            for seq in (sequences if isinstance(sequences, list) and isinstance(sequences[0], list) else [sequences]):
                pre_weights.append([seq.count(idseq) for idseq in range(self.num_operators)])

            multipliers = np.exp(-np.argsort([y[-1] for y in (
                fitness_values if isinstance(fitness_values, list) and isinstance(fitness_values[0], list) else
                [fitness_values])])).reshape((len(fitness_values), 1))
            weights = np.sum(np.exp(np.array(pre_weights)) * multipliers, axis=0)
            self.weights_array = weights / weights.sum()


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


class HyperheuristicError(Exception):
    """
    Simple HyperheuristicError to manage exceptions.
    """
    pass


if __name__ == '__main__':
    # import hyperheuristic as hh
    import benchmark_func as bf
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    # problem = bf.Sphere(50)
    problem = bf.Stochastic(50)
    problem.set_search_range(-10, 10)

    q = Hyperheuristic(problem=problem.get_formatted_problem(),
                       file_label="{}-{}D".format(problem.func_name, problem.variable_num))
    q.parameters['num_steps'] = 100
    q.parameters['stagnation_percentage'] = 1
    q.parameters['num_replicas'] = 50
    fitprep, seqrep = q.solve('dynamic')

    colours = plt.cm.hot_r(np.linspace(0, 1, len(fitprep)))
    fi = plt.figure()
    plt.ion()
    for x, c in zip(fitprep, colours):
        plt.plot(x, '-o', color=c)
    plt.ioff()
    # plt.plot(c, 'o')
    fi.show()

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    plt.ion()
    for x, y, c in zip(fitprep, seqrep, colours):
        ax.plot3D(range(len(x)), y, x, 'o-', color=c)

    plt.ioff()
    plt.show()

    # import numpy as np
    #
    # qq, _ = q._obtain_candidate_solution(np.array(range(2)), 'RemoveMany')
