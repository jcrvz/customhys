# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 15:36:43 2020

@author: Jorge Mario Cruz-Duarte (jcrvz.github.io)
"""
import numpy as np
import scipy.stats as st
from metaheuristic import Metaheuristic


class Hyperheuristic():
    def __init__(self, heuristic_space, problem, parameters={
            'cardinality': 2, 'num_iterations': 100, 'num_agents': 30,
            'num_replicas': 100}):
        # Read the heuristic space (mandatory)
        self.heuristic_space = heuristic_space
        self.problem = problem
        self.num_operators = len(heuristic_space)

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
            A dictionary of information from each iteration. Its keys are:
            'iteration', 'encoded_solution', 'solution', 'performances', and
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

        # Historical variables
        historicals = dict(
            iteration=[0],
            encoded_solution=[encoded_solution],
            solution=[solution],
            performances=[performance],
            details=[details])

        # Perform the random search
        for iteration in range(self.parameters['num_iterations']):

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

                # Update historicals (only if improves)
                historicals['iteration'].append(iteration)
                historicals['encoded_solution'].append(encoded_solution)
                historicals['solution'].append(solution)
                historicals['performance'].append(performance)
                historicals['details'].append(candidate_details)

        return solution, performance, encoded_solution, historicals

    def evaluate_metaheuristic(self, search_operators):
        # Call the metaheuristic
        mh = Metaheuristic(self.problem, search_operators,
                           self.parameters['num_agents'])

        # Initialise the historical registers
        historical_data = list()
        fitness_data = list()
        position_data = list()

        # Run the metaheuristic several times
        for rep in range(self.parameters['num_replicas']):
            # Run this metaheuristic
            mh.run()

            # Store the historical values from this run
            historical_data.append(mh.historical)

            # Read and store the solution obtained
            _temporal_position, _temporal_fitness = mh.get_solution()
            fitness_data.append(_temporal_position)
            position_data.append(_temporal_fitness)

        # Determine a performance metric once finish the repetitions
        fitness_stats = self.get_statistics(fitness_data)

        return self.get_performance(fitness_stats), dict(
            historical=historical_data, fitness=fitness_data,
            positions=position_data, statistics=fitness_stats)

    def get_performance(self, statistics):
        # Score function using the statistics from fitness values
        # perf = statistics['Avg']  # Option 1
        # perf = statistics['Avg'] + statistics['Std']  # Option 2
        # perf = statistics['Med'] + statistics['IQR']  # Option 3
        perf = statistics['Avg'] + statistics['Std'] + \
            statistics['Med'] + statistics['IQR']
        return perf

    def get_statistics(self, raw_data):
        # Get descriptive statistics
        dst = st.describe(raw_data)

    # Store statistics
        return dict(nob=dst['nobs'],
                    Min=dst['minmax'][0],
                    Max=dst['minmax'][1],
                    Avg=dst['mean'],
                    Std=np.std(raw_data),
                    Skw=dst['skewness'],
                    Kur=dst['kurtosis'],
                    IQR=st.iqr(raw_data),
                    Med=np.median(raw_data),
                    MAD=st.median_absolute_deviation(raw_data))
