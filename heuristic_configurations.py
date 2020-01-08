# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:09:11 2019

@author: L03130342
"""
from numpy import linspace, size, prod, meshgrid, arange


def get_search_operators(num_vals=5):
    """
    Generate a list of all the available search operators with a given
    number of values for each parameter (if so).

    Parameters
    ----------
    num_vals : int, optional
        Number of values to generate per each numerical paramenter in
        a search operator. The default is 5.

    Returns
    -------
    list
        A list of all the available search operators. Each element of this
        list has the following structure:

            search_operator = ("name_of_the_operator",
                               dict(parameter1=[value1, value2, ...],
                                    parameter2=[value2, value2, ...],
                                    ...),
                               "name_of_the_selector")
    """
    return [
        (
            "local_random_walk",
            dict(
                probability=linspace(0.0, 1.0, num_vals),
                scale=linspace(0.0, 1.0, num_vals)),
            "greedy"),
        (
            "random_search",
            dict(
                scale=linspace(0.0, 1.0, num_vals)),
            "greedy"),
        (
            "random_sample",
            dict(),
            "greedy"),
        (
            "rayleigh_flight",
            dict(
                scale=linspace(0.0, 1.0, num_vals)),
            "greedy"),
        (
            "levy_flight",
            dict(
                scale=linspace(0.0, 1.0, num_vals),
                beta=[1.5]),
            "greedy"),
        (
            "differential_mutation",
            dict(
                expression=["rand", "best", "current", "current-to-best",
                            "rand-to-best", "rand-to-best-and-current"],
                num_rands=[1, 2, 3],
                factor=linspace(0.0, 2.0, num_vals)),
            "greedy"),
        (
            'differential_crossover',
            dict(
                crossover_rate=linspace(0.0, 1.0, num_vals),
                version=["binomial", "exponential"]),
            "greedy"),
        (
            "firefly_dynamic",
            dict(
                epsilon=["uniform", "gaussian"],
                alpha=linspace(0.0, 1.0, num_vals),
                beta=[1.0],
                gamma=linspace(1.0, 100.0, num_vals)),
            "greedy"),
        (
            "swarm_dynamic",
            dict(
                factor=linspace(0.0, 1.0, num_vals),
                self_conf=linspace(0.0, 5.0, num_vals),
                swarm_conf=linspace(0.0, 5.0, num_vals),
                version=["inertial", "constriction"]),
            "all"),
        (
            "gravitational_search",
            dict(
                gravity=linspace(0.0, 1.0, num_vals),
                alpha=linspace(0.0, 1.0, num_vals),
                epsilon=[1e-23]),
            "all"),
        (
            "central_force_dynamic",
            dict(
                gravity=linspace(0.0, 1.0, num_vals),
                alpha=linspace(0.0, 1.0, num_vals),
                beta=linspace(0.0, 3.0, num_vals),
                dt=[1.0]),
            "all"),
        (
            "spiral_dynamic",
            dict(
                radius=linspace(0.0, 1.0, num_vals),
                angle=linspace(0.0, 180.0, num_vals),
                sigma=linspace(0.0, 0.5, num_vals)),
            "all"),
        (
            "genetic_mutation",
            dict(
                elite_rate=linspace(0.0, 1.0, num_vals),
                mutation_rate=linspace(0.0, 1.0, num_vals),
                distribution=["uniform", "gaussian"],
                sigma=linspace(0.0, 1.0, num_vals)),
            "all"),
        (
            "genetic_crossover",
            dict(
                pairing=["even-odd", "rank", "cost", "random",
                         "tournament_2_100", "tournament_2_75",
                         "tournament_2_50", "tournament_3_100",
                         "tournament_3_75", "tournament_3_50"],
                crossover=["single", "two", "uniform", "blend",
                           "linear_0.5_0.5", "linear_1.5_0.5",
                           "linear_0.5_1.5", "linear_1.5_1.5",
                           "linear_-0.5_0.5", "linear_0.5_-0.5"],
                mating_pool_factor=linspace(0.0, 1.0, num_vals)),
            "all")
        ]


def generate_heuristics(heuristics=get_search_operators()):
    # Counters: [classes, methods]
    total_counters = [0, 0]

    # Initialise the collection of simple heuristics
    file = open('operators_collection.txt', 'w')

    # For each simple heuristic, read their parameters and values
    for operator, parameters, selector in heuristics:
        # Update the total classes counter
        total_counters[0] += 1

        # Read the number of parameters and how many values have each one
        num_parameters = len(parameters)
        if num_parameters > 0:
            # Read the name and possible values of parameters
            par_names = list(parameters.keys())
            par_values = list(parameters.values())

            # Find the number of values for each parameter
            par_num_values = [size(x) for x in par_values]

            # Determine the number of combinations
            num_combinations = prod(par_num_values)

            # Create the table of all possible combinations (index/parameter)
            indices = [x.flatten() for x in meshgrid(
                *list(map(lambda x: arange(x), par_num_values)))]

            # For each combination, create a single dictionary which
            # corresponds to a simple search operator
            for combi in range(num_combinations):
                list_tuples = [(par_names[k],
                                par_values[k][indices[k][combi]])
                               for k in range(num_parameters)]
                simple_par_combination = dict(list_tuples)
                file.write("('{}', {}, '{}')\n".format(
                    operator, simple_par_combination, selector))
        else:
            num_combinations = 0

        # Update the total combination counter
        total_counters[1] += num_combinations

        print(f"{operator}: parameters={num_parameters}, " +
              f"combinations:{num_combinations}")

    file.close()
    print("-" * 50 + "--\nTOTAL: classes=%d, operators=%d" %
          tuple(total_counters))


# ----------------------------------------------------------------------------
# Set the number of values per parameter
# num_values = 5

# Generate a list for testing purposes (using five values / parameter)
# search_operators = get_search_operators(num_values)

# Create the file of heuristics
# generate_heuristics(search_operators)
