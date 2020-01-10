# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 15:36:43 2020

@author: Jorge Mario Cruz-Duarte (jcrvz.github.io)
"""
import numpy as np


# TODO: implement the decoding method
def random_search(heuristic_space, cardinality, num_iterations=100):
    # Read the number of search operators
    num_operators = len(heuristic_space)

    # Create the initial solution
    encoded_solution = np.random.randint(0, num_operators, cardinality)
    solution = [heuristic_space[index] for index in encoded_solution]

    # Evaluate this solution
    performance = evaluate_metaheuristic(solution)

    # Historical variables
    historical = dict(
        encoded_solution=[encoded_solution],
        solution=[solution],
        performances=[performance])

    # Perform the random search
    for iteration in range(num_iterations):

        # Select randomly a candidate solution
        encoded_candidate_solution = np.random.randint(0, num_operators,
                                                       cardinality)
        candidate_solution = [heuristic_space[index] for index in
                              encoded_candidate_solution]

        # Evaluate this candidate solution
        candidate_performance = evaluate_metaheuristic(
            candidate_solution)

        # Check improvement (greedy selection)
        if candidate_performance < performance:
            encoded_solution = encoded_candidate_solution
            solution = candidate_solution
            performance = candidate_performance

        # Update historicals
        historical['encoded_solution'].append(encoded_solution)
        historical['solution'].append(solution)
        historical['performance'].append(performance)
