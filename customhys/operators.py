# -*- coding: utf-8 -*-
"""
This module contains a collection search operators extracted from several well-known metaheuristics from the literature.
All the available operators are listed in ``__all__``

Created on Tue Jan  7 14:54:31 2020

@author: Jorge Mario Cruz-Duarte (jcrvz.github.io), e-mail: jorge.cruz@tec.mx
"""

import numpy as np
import os
from itertools import combinations as _get_combinations
from population import __selectors__

__all__ = ['local_random_walk', 'random_search', 'random_sample', 'random_flight', 'differential_mutation',
           'firefly_dynamic', 'swarm_dynamic', 'gravitational_search', 'central_force_dynamic', 'spiral_dynamic',
           'genetic_mutation', 'genetic_crossover']


# Search operator aliases
def get_operator_aliases():
    """
    Return two dictionaries with the perturbator and selector aliases for better naming the metaheuristics
    @return: dict, dict
    """
    return {
        'random_search': 'RS',
        'central_force_dynamic': 'CF',
        'differential_mutation': 'DM',
        'firefly_dynamic': 'FD',
        'genetic_crossover': 'GC',
        'genetic_mutation': 'GM',
        'gravitational_search': 'GS',
        'random_flight': 'RF',
        'local_random_walk': 'RW',
        'random_sample': 'RX',
        'spiral_dynamic': 'SD',
        'swarm_dynamic': 'PS'}, {
        'greedy': 'g', 'all': 'd', 'metropolis': 'm', 'probabilistic': 'p'}

# %% SEARCH OPERATORS

def central_force_dynamic(pop, gravity=0.001, alpha=0.01, beta=1.5, dt=1.0):
    """
    Apply the central force dynamic from Central Force Optimisation (CFO) to the population's positions (pop.positions).

    :param population pop: population.
    :param float gravity: optional.
        It is the gravitational constant. The default is 0.001.
    :param float alpha: optional.
        It is the power mass factor. The default is 0.01.
    :param float beta: optional.
        It is the power distance factor. The default is 1.5.
    :param float dt: optional.
        It is the time interval between steps. The default is 1.0.

    :return: None.
    """
    # Initialise acceleration
    acceleration = np.zeros((pop.num_agents, pop.num_dimensions))

    for agent in range(pop.num_agents):
        # Select indices in order to avoid division by zero
        indices = (np.arange(pop.num_agents) != agent)

        # Determine all masses differences with respect to agent
        delta_masses = pop.fitness[indices] - np.tile(
            pop.fitness[agent], (1, pop.num_agents - 1))

        # Determine all vector distances with respect to agent
        delta_positions = pop.positions[indices, :] - np.tile(pop.positions[agent, :], (pop.num_agents - 1, 1))

        distances = np.linalg.norm(delta_positions, 2, 1)

        # Find the quotient part
        quotient = np.heaviside(-delta_masses, 0.0) * (np.abs(delta_masses) ** alpha) / (distances ** beta + 1e-23)

        # Determine the acceleration for each agent
        acceleration[agent, :] = gravity * np.sum(delta_positions * np.tile(
            quotient.transpose(), (1, pop.num_dimensions)), 0)

    pop.positions += 0.5 * acceleration * (dt ** 2)


def differential_crossover(pop, crossover_rate=0.2, version='binomial'):
    """
    Apply the differential crossover from Differential Evolution (DE) to the population's positions (pop.positions).

    :param population pop: population.
        It is a population object.
    :param float crossover_rate: optional.
        Probability factor to perform the crossover. The default is 0.2.
    :param str version: optional.
        Crossover version. It can be 'binomial' or 'exponential'. The default is 'binomial'.

    :return: None.
    """

    # Binomial version
    if version == 'binomial':
        # Define indices
        indices = np.tile(np.arange(pop.num_dimensions), (pop.num_agents, 1))

        # Permute indices per dimension
        rand_indices = np.vectorize(np.random.permutation, signature='(n)->(n)')(indices)

        # Calculate the NOT condition (because positions already updated!)
        condition = np.logical_not(
            (indices == rand_indices) | (np.random.rand(pop.num_agents, pop.num_dimensions) <= crossover_rate))

        # Reverse the ones to their previous positions
        pop.positions[condition] = np.copy(pop.previous_positions[condition])

    # Exponential version
    elif version == 'exponential':
        # Perform the exponential crossover procedure
        for agent in range(pop.num_agents):
            for dim in range(pop.num_dimensions):
                # Initialise L and choose a random index n
                exp_var = 0
                n = np.random.randint(pop.num_dimensions)
                while True:
                    # Increase L and check the exponential CR condition
                    exp_var += 1
                    if np.logical_not((np.random.rand() < crossover_rate) and (exp_var < pop.num_dimensions)):
                        break

                # Perform the crossover if the following condition is met
                if dim not in [(n + x) % pop.num_dimensions for x in range(exp_var)]:
                    pop.positions[agent, dim] = np.copy(pop.previous_positions[agent, dim])

    # Invalid version
    else:
        raise OperatorsError('Invalid differential_crossover version')


def differential_mutation(pop, expression='current-to-best', num_rands=1, factor=1.0):
    """
    Apply the differential mutation from Differential Evolution (DE) to the population's positions (pop.positions).

    :param population pop: population.
        It is a population object.
    :param str expression: optional.
        Type of DE mutation. Available mutations: 'rand', 'best', 'current', 'current-to-best', 'rand-to-best',
         'rand-to-best-and-current'. The default is 'current-to-best'.
    :param int num_rands: optional.
        Number of differences between positions selected at random. The default is 1.
    :param float factor: optional.
        Scale factor (F) to weight contributions from other agents. The default is 1.0.

    :return: None.

    """
    # TODO: Include the expression 'current-to-pbest'
    # Create mutants using the expression provided in scheme
    if expression == 'rand':
        mutant = pop.positions[np.random.permutation(pop.num_agents), :]

    elif expression == 'best':
        mutant = np.tile(pop.global_best_position, (pop.num_agents, 1))

    elif expression == 'current':
        mutant = pop.positions

    elif expression == 'current-to-best':
        mutant = pop.positions + factor * (np.tile(pop.global_best_position, (pop.num_agents, 1)) -
                                           pop.positions[np.random.permutation(pop.num_agents), :])

    elif expression == 'rand-to-best':
        mutant = pop.positions[np.random.permutation(pop.num_agents), :] + factor * (np.tile(
            pop.global_best_position, (pop.num_agents, 1)) - pop.positions[np.random.permutation(pop.num_agents), :])

    elif expression == 'rand-to-best-and-current':
        mutant = pop.positions[np.random.permutation(pop.num_agents), :] + factor * (
                np.tile(pop.global_best_position, (pop.num_agents, 1)) -
                pop.positions[np.random.permutation(pop.num_agents), :] +
                pop.positions[np.random.permutation(pop.num_agents), :] - pop.positions)
    else:
        raise OperatorsError('Invalid DE mutation scheme!')

    # Add random parts according to num_rands
    if num_rands >= 0:
        for _ in range(num_rands):
            mutant += factor * (pop.positions[np.random.permutation(pop.num_agents), :] -
                                pop.positions[np.random.permutation(pop.num_agents), :])
    else:
        raise OperatorsError('Invalid DE mutation scheme!')

    # Replace mutant population in the current one
    pop.positions = np.copy(mutant)


def firefly_dynamic(pop, alpha=1.0, beta=1.0, gamma=100.0, distribution='uniform'):
    """
    Apply the firefly dynamic from Firefly algorithm (FA) to the population's positions (pop.positions).

    :param population pop: population.
        It is a population object.
    :param float alpha: optional.
        Scale of the random value. The default is 1.0.
    :param float beta: optional.
        Scale of the firefly contribution. The default is 1.0.
    :param float gamma: optional.
        Light damping parameters. The default is 100.
    :param str distribution: optional.
        Type of random number. Possible options: 'gaussian', 'uniform', and 'levy'. The default is 'uniform'.

    :return: None.
    """
    # Determine epsilon values
    if distribution == 'gaussian':
        epsilon_value = np.random.standard_normal((pop.num_agents, pop.num_dimensions))

    elif distribution == 'uniform':
        epsilon_value = np.random.uniform(-0.5, 0.5, (pop.num_agents, pop.num_dimensions))
    elif distribution == 'levy':
        epsilon_value = _random_levy((pop.num_agents, pop.num_dimensions), 1.5)
    else:
        raise OperatorsError('Invalid distribution')

    # Initialise delta or difference between two positions
    difference_positions = np.zeros((pop.num_agents, pop.num_dimensions))

    for agent in range(pop.num_agents):
        # Select indices in order to avoid division by zero
        indices = (np.arange(pop.num_agents) != agent)

        # Determine all vector distances with respect to agent
        delta = pop.positions[indices, :] - np.tile(pop.positions[agent, :], (pop.num_agents - 1, 1))

        # Determine differences between lights
        delta_lights = np.tile((pop.fitness[indices] - np.tile(
            pop.fitness[agent], (1, pop.num_agents - 1))).transpose(), (1, pop.num_dimensions))

        # Find the total attraction for each agent
        difference_positions[agent, :] = np.sum(np.heaviside(-delta_lights, 0.0) * delta * np.exp(-gamma * np.tile(
            np.linalg.norm(delta, 2, 1).reshape(pop.num_agents - 1, 1), (1, pop.num_dimensions)) ** 2), 0)

    # Move fireflies according to their attractions
    pop.positions += alpha * epsilon_value + beta * difference_positions


def genetic_crossover(pop, pairing='rank', crossover='blend', mating_pool_factor=0.4):
    """
    Apply the genetic crossover from Genetic Algorithm (GA) to the population's positions (pop.positions).

    :param population pop: population.
        It is a population object.
    :param str pairing: optional.
        It indicates which pairing scheme to employ. Pairing schemes available are: 'cost' (Roulette Wheel or
        Cost Weighting), 'rank' (Rank Weighting), 'tournament', 'random', and 'even-odd'.
        When tournament is chosen, tournament size (tp) and probability (tp) can be encoded such as
        'tournament_{ts}_{tp}', {ts} and {tp}. Writing only 'tournament' is similar to specify 'tournament_3_100'.
        The default is 'rank'.
    :param str crossover: optional.
        It indicates which crossover scheme to employ. Crossover schemes available are: 'single', 'two', 'uniform',
        'blend', and 'linear'. Likewise 'tournament' pairing, coefficients of 'linear' can be enconded such as
        'linear_{coeff1}_{coeff2}' where the offspring is determined as follows:
            ``offspring = coeff1 * father + coeff2 * mother``
        The default is 'blend'.
    :param float mating_pool_factor: optional.
        It indicates the proportion of population to disregard. The default is 0.4.

    :return: None.
    """
    # Mating pool size
    num_mates = int(np.round(mating_pool_factor * pop.num_agents))

    # Get parents (at least a couple per offspring)
    if len(pairing) > 10:  # if pairing = 'tournament_2_100', for example
        pairing, tournament_size, tournament_probability = pairing.split("_")
        tournament_size = int(tournament_size)
        if num_mates < tournament_size:
            num_mates = tournament_size
    else:  # dummy (it must not be used)
        tournament_size, tournament_probability = '3', '100'

    # Number of offsprings (or couples)
    num_couples = pop.num_agents - num_mates

    # Get the mating pool using the natural selection
    mating_pool_indices = np.argsort(pop.fitness)[:num_mates]
    #
    # Roulette Wheel (Cost Weighting) Selection
    if pairing == 'cost':
        # Cost normalisation from mating pool: cost-min(cost @ non mates)
        normalised_cost = pop.fitness[mating_pool_indices] - np.min(
            pop.fitness[np.setdiff1d(np.arange(pop.num_agents), mating_pool_indices)])

        # Determine the related probabilities
        probabilities = np.abs(normalised_cost / (np.sum(normalised_cost) + 1e-23))

        # Perform the roulette wheel selection and return couples
        couple_indices_ = np.searchsorted(np.cumsum(probabilities), np.random.rand(2 * num_couples))

        # Return couples
        couple_indices = couple_indices_.reshape((2, -1))

    # Roulette Wheel (Rank Weighting) Selection
    elif pairing == 'rank':
        # Determine the probabilities
        probabilities = (mating_pool_indices.size - np.arange(
            mating_pool_indices.size)) / np.sum(np.arange(mating_pool_indices.size) + 1)

        # Perform the roulette wheel selection and return couples
        couple_indices_ = np.searchsorted(np.cumsum(probabilities), np.random.rand(2 * num_couples))

        # Return couples
        couple_indices = couple_indices_.reshape((2, -1))

    # Tournament pairing
    elif pairing == 'tournament':
        # Calculate probabilities
        probability = float(tournament_probability) / 100.
        probabilities = probability * ((1 - probability) ** np.arange(tournament_size))

        # Initialise the mother and father indices
        couple_indices = np.full((2, num_couples), np.nan)

        # Perform tournaments until all mates are selected
        for couple in range(num_couples):
            mate = 0
            while mate < 2:
                # Choose tournament candidates
                random_indices = mating_pool_indices[np.random.permutation(mating_pool_indices.size)[:tournament_size]]

                # Determine the candidate fitness values
                candidates_indices = random_indices[np.argsort(pop.fitness[random_indices])]

                # Find the best according to its fitness and probability
                winner = candidates_indices[np.random.rand(tournament_size) < probabilities]
                if winner.size > 0:
                    couple_indices[mate, couple] = int(winner[0])
                    mate += 1

    # Random pairing
    elif pairing == 'random':
        # Return two random indices from mating pool
        couple_indices = mating_pool_indices[np.random.randint(mating_pool_indices.size, size=(2, num_couples))]

    # TODO: Check Even-and-Odd pairing
    # Even-and-Odd pairing
    # elif pairing == "even-odd":
    #     # Check if the num of mates is even
    #     mating_pool_size = mating_pool_indices.size - \
    #         (mating_pool_indices.size % 2)
    #     half_size = mating_pool_size // 2
    #
    #     # Dummy indices according to the mating pool size
    #     remaining = num_couples - half_size
    #     if remaining > 0:
    #         dummy_indices = np.tile(
    #             np.reshape(np.arange(mating_pool_size),
    #                        (-1, 2)).transpose(),
    #             (1, int(np.ceil(num_couples / half_size))))
    #     else:
    #         dummy_indices = np.reshape(np.arange(mating_pool_size),
    #                                    (-1, 2)).transpose()
    #
    #     # Return couple_indices
    #     couple_indices = mating_pool_indices[
    #         dummy_indices[:, :num_couples]]

    # If no pairing procedure recognised
    else:
        raise OperatorsError("Invalid pairing method")

    # Identify offspring indices
    offspring_indices = np.setdiff1d(np.arange(pop.num_agents), mating_pool_indices, True)

    # Prepare crossover variables
    if len(crossover) > 7:  # if crossover = 'linear_0.5_0.5', for example
        cr_split = crossover.split("_")
        if len(cr_split) == 1:
            crossover = cr_split
            coeff1 = coeff2 = 0.5
        elif len(cr_split) == 2:
            crossover = cr_split[0]
            coeff1 = coeff2 = cr_split[1]
        else:
            crossover = cr_split[0]
            coeff1 = cr_split[1]
            coeff2 = cr_split[2]
        coefficients = [float(coeff1), float(coeff2)]
    else:  # dummy (it must not be used)
        coefficients = [np.nan, np.nan]

    # Perform crossover and assign to population
    parent_indices = couple_indices.astype(np.int64)

    # Single-Point Crossover
    if crossover == 'single':
        # Determine the single point per each couple
        single_points = np.tile(np.random.randint(
            pop.num_dimensions, size=parent_indices.shape[1]), (pop.num_dimensions, 1)).transpose()

        # Crossover condition mask
        crossover_mask = np.tile(np.arange(pop.num_dimensions), (parent_indices.shape[1], 1)) <= single_points

        # Get father and mother
        father_position = pop.positions[parent_indices[0, :], :]
        mother_position = pop.positions[parent_indices[1, :], :]

        # Initialise offsprings with mother positions
        offsprings = mother_position
        offsprings[crossover_mask] = father_position[crossover_mask]

    # Two-Point Crossover
    elif crossover == 'two':
        # Find raw points
        raw_points = np.sort(np.random.randint(pop.num_dimensions, size=(parent_indices.shape[1], 2)))

        # Determine the single point per each couple
        points = [np.tile(raw_points[:, x], (pop.num_dimensions, 1)).transpose() for x in range(raw_points.shape[1])]

        # Range matrix
        dummy_matrix = np.tile(np.arange(pop.num_dimensions), (parent_indices.shape[1], 1))

        # Crossover condition mask (only for two points)
        crossover_mask = np.bitwise_or(dummy_matrix <= points[0], dummy_matrix > points[1])

        # Get father and mother
        father_position = pop.positions[parent_indices[0, :], :]
        mother_position = pop.positions[parent_indices[1, :], :]

        # Initialise offsprings with mother positions
        offsprings = mother_position
        offsprings[crossover_mask] = father_position[crossover_mask]

    # Uniform Crossover
    elif crossover == 'uniform':
        # Crossover condition mask (only for uniform crossover)
        crossover_mask = np.random.rand(parent_indices.shape[1], pop.num_dimensions) < 0.5

        # Get father and mother
        father_position = pop.positions[parent_indices[0, :], :]
        mother_position = pop.positions[parent_indices[1, :], :]

        # Initialise offsprings with mother positions
        offsprings = mother_position
        offsprings[crossover_mask] = father_position[crossover_mask]

    # Random blending crossover
    elif crossover == 'blend':
        # Initialise random numbers between 0 and 1
        beta_values = np.random.rand(parent_indices.shape[1], pop.num_dimensions)

        # Get father and mother
        father_position = pop.positions[parent_indices[0, :], :]
        mother_position = pop.positions[parent_indices[1, :], :]

        # Determine offsprings with father and mother positions
        offsprings = beta_values * father_position + (1 - beta_values) * mother_position

    # Linear Crossover: offspring = coeff[0] * father + coeff[1] * mother
    elif crossover == 'linear':
        # Get father and mother
        father_position = pop.positions[parent_indices[0, :], :]
        mother_position = pop.positions[parent_indices[1, :], :]

        # Determine offsprings with father and mother positions
        offsprings = coefficients[0] * father_position + coefficients[1] * mother_position

    # If no crossover method recognised
    else:
        raise OperatorsError('Invalid pairing method')

    # Store offspring positions in the current population
    pop.positions[offspring_indices, :] = np.copy(offsprings)


def genetic_mutation(pop, scale=1.0, elite_rate=0.1, mutation_rate=0.25, distribution='uniform'):
    """
    Apply the genetic mutation from Genetic Algorithm (GA) to the population's positions (pop.positions).

    :param population pop: population.
        It is a population object.
    :param float scale: optional.
        It is the scale factor of the mutations. The default is 1.0.
    :param float elite_rate : optional.
        It is the proportion of population to preserve. The default is 0.1.
    :param float mutation_rate: optional.
        It is the proportion of population to mutate. The default is 0.25.
    :param str distribution: optional.
        It indicates the random distribution that power the mutation. There are only two distribution available:
        'uniform', 'gaussian', and 'levy'. The default is 'uniform'.
    :return: None.
    """

    # Calculate the number of elite agents
    num_elite = int(np.ceil(pop.num_agents * elite_rate))

    # If num_elite equals num_agents then do nothing, or ...
    if num_elite < pop.num_agents:
        # Number of mutations to perform
        num_mutations = int(np.round(pop.num_agents * pop.num_dimensions * mutation_rate))

        # Identify mutable agents
        dimension_indices = np.random.randint(0, pop.num_dimensions, num_mutations)

        if num_elite > 0:
            agent_indices = np.argsort(pop.fitness)[np.random.randint(num_elite, pop.num_agents, num_mutations)]
        else:
            agent_indices = np.random.randint(num_elite, pop.num_agents, num_mutations)

        # Transform indices
        rows, columns = np.meshgrid(agent_indices, dimension_indices)

        # Perform mutation according to the random distribution
        if distribution == 'uniform':
            mutants = np.random.uniform(-1, 1, num_mutations ** 2)

        elif distribution == 'gaussian':
            # Normal with mu = 0 and sigma = parameter
            mutants = np.random.standard_normal(num_mutations ** 2)

        elif distribution == 'levy':
            mutants = _random_levy(num_mutations ** 2, 1.5)

        else:
            raise OperatorsError('Invalid distribution!')

        # Store mutants
        pop.positions[rows.flatten(), columns.flatten()] = scale * mutants


def gravitational_search(pop, gravity=1.0, alpha=0.02):
    """
    Apply the gravitational search from Gravitational Search Algorithm (GSA) to the population's positions
    (pop.positions).

    :param population pop : population.
        It is a population object.
    :param float gravity: optional.
        It is the initial gravitational value. The default is 1.0.
    :param float alpha: optional.
        It is the gravitational damping ratio. The default is 0.02.

    :return: None.
    """

    # Initialise acceleration
    acceleration = np.zeros((pop.num_agents, pop.num_dimensions))

    # Determine the gravitational constant
    gravitation = gravity * np.exp(- alpha * pop.iteration)

    # Determine mass for each agent
    raw_masses = (pop.fitness - np.tile(pop.current_worst_fitness, (1, pop.num_agents)))
    masses = (raw_masses / (np.sum(raw_masses) + 1e-23)).reshape(pop.num_agents)

    for agent in range(pop.num_agents):
        # Select indices in order to avoid division by zero
        indices = (np.arange(pop.num_agents) != agent)

        # Determine all vector distances with respect to agent
        delta_positions = pop.positions[indices, :] - np.tile(pop.positions[agent, :], (pop.num_agents - 1, 1))

        quotient = masses[indices] / (np.linalg.norm(delta_positions, 2, 1) + 1e-23)

        # Force interaction
        force_interaction = gravitation * np.tile(
            quotient.reshape(pop.num_agents - 1, 1), (1, pop.num_dimensions)) * delta_positions

        # Acceleration
        acceleration[agent, :] = np.sum(np.random.rand(pop.num_agents - 1, pop.num_dimensions) * force_interaction, 0)

    # Update velocities
    # TODO: Add different random distributions
    pop.velocities = acceleration + np.random.rand(pop.num_agents, pop.num_dimensions) * pop.velocities

    # Update positions
    pop.positions += pop.velocities


def random_flight(pop, scale=1.0, distribution='levy', beta=1.5):
    """
    Apply the random flight from Random Search (RS) to the population's positions (pop.positions).

    :param population pop : population.
        It is a population object.
    :param float scale: optional.
        It is the step scale. The default is 1.0.
    :param str distribution: optional.
        It is the distribution to draw the random samples. The default is 'levy'.
    :param float beta: optional
        It is the distribution parameter between [1.0, 3.0]. This paramenter only has sense when distribution='levy'.
         The default is 1.5.

    :return: None.
    """

    # Get random samples
    if distribution == 'uniform':
        random_samples = np.random.uniform(
            size=(pop.num_agents, pop.num_dimensions))

    elif distribution == 'gaussian':
        # Normal with mu = 0 and sigma = parameter
        random_samples = np.random.standard_normal(
            (pop.num_agents, pop.num_dimensions))

    elif distribution == 'levy':
        # Calculate the random number with levy stable distribution
        random_samples = _random_levy(size=(pop.num_agents, pop.num_dimensions), beta=beta)

    else:
        raise OperatorsError('Invalid distribution!')

    # Move each agent using levy random displacements
    pop.positions += scale * random_samples * (pop.positions - np.tile(pop.global_best_position, (pop.num_agents, 1)))


def local_random_walk(pop, probability=0.75, scale=1.0, distribution='uniform'):
    """
    Apply the local random walk from Cuckoo Search (CS) to the population's positions (pop.positions).

    :param population pop: population.
        It is a population object.
    :param float probability: optional.
        It is the probability of discovering an alien egg (change an agent's position). The default is 0.75.
    :param float scale: optional.
        It is the step scale. The default is 1.0.
    :param str distribution: optional.
        It is the random distribution used to sample the stochastic variable. The default value is 'uniform'.

    :return: None.
    """

    # Determine random numbers
    if distribution == "uniform":
        r_1 = np.random.rand(pop.num_agents, pop.num_dimensions)
    elif distribution == "gaussian":
        r_1 = np.random.randn(pop.num_agents, pop.num_dimensions)
    elif distribution == "levy":
        r_1 = _random_levy(size=(pop.num_agents, pop.num_dimensions))
    else:
        raise OperatorsError('Invalid distribution!')
    r_2 = np.random.rand(pop.num_agents, pop.num_dimensions)

    # Move positions with a displacement due permutations and probabilities
    pop.positions += scale * r_1 * (pop.positions[
                                    np.random.permutation(pop.num_agents), :] - pop.positions[
                                                                                np.random.permutation(pop.num_agents),
                                                                                :]) * np.heaviside(r_2 - probability,
                                                                                                   0.0)


def random_sample(pop):
    """
    Apply the random_sample to the population's positions (pop.positions). This operator has no memory.

    :param population pop: population.
        It is a population object.

    :return: None.
    """
    # Create random positions using random numbers between -1 and 1
    pop.positions = np.random.uniform(-1, 1, (pop.num_agents, pop.num_dimensions))


def random_search(pop, scale=0.01, distribution='uniform'):
    """
    Apply the random search from Random Search (RS) to the population's positions (pop.positions).

    :param population pop : population.
        It is a population object.
    :param float scale: optional.
        It is the step scale. The default is 0.01.
    :param str distribution: optional.
        It is the distribution used to perform the random search. The default is 'uniform'.

    :return: None.
    """
    # Determine the random step
    if distribution == "uniform":
        random_step = np.random.uniform(-1, 1, (pop.num_agents, pop.num_dimensions))
    elif distribution == "gaussian":
        random_step = np.random.standard_normal((pop.num_agents, pop.num_dimensions))
    elif distribution == "levy":
        random_step = _random_levy(size=(pop.num_agents, pop.num_dimensions))
    else:
        raise OperatorsError('Invalid distribution!')

    # Move each agent using uniform random displacements
    pop.positions += scale * random_step


def spiral_dynamic(pop, radius=0.9, angle=22.5, sigma=0.1):
    """
    Apply the spiral dynamic from Stochastic Spiral Optimisation (SSO) to the population's positions (pop.positions).

    :param population pop: population.
        It is a population object.
    :param float radius: optional.
        It is the convergence rate. The default is 0.9.
    :param float angle: optional.
        Rotation angle (in degrees). The default is 22.5 (degrees).
    :param float sigma: optional.
        Variation of random radii. The default is 0.1.
        Note: if sigma equals 0.0, the operator corresponds to that from the Deterministic Spiral Algorithm.

    :return: None.
    """
    # Determine the rotation matrix
    rotation_matrix = get_rotation_matrix(pop.num_dimensions, np.deg2rad(angle))

    for agent in range(pop.num_agents):
        random_radii = np.random.uniform(radius - sigma, radius + sigma, pop.num_dimensions)
        # If random radii need to be constrained to [0, 1]:
        pop.positions[agent, :] = pop.global_best_position + random_radii * \
                                  np.matmul(rotation_matrix, (pop.positions[agent, :] - pop.global_best_position))


def swarm_dynamic(pop, factor=1.0, self_conf=2.54, swarm_conf=2.56, version='constriction', distribution='uniform'):
    """
    Apply the swarm dynamic from Particle Swarm Optimisation (PSO) to the population's positions (pop.positions).

    :param population pop: population.
        It is a population object.
    :param float factor: optional.
        Inertial or Kappa factor, depending of which PSO version is set. The default is 1.0.
    :param float self_conf: optional.
        Self confidence factor. The default is 2.54.
    :param float swarm_conf: optional.
        Swarm confidence factor. The default is 2.56.
    :param str version: optional.
        Version of the Particle Swarm Optimisation strategy. It can be 'constriction' or 'inertial'. The default is
        'constriction'.
    :param str distribution: optional.
        Distribution to draw the random numbers. It can be 'uniform', 'gaussian', and 'levy'.

    :return: None.
    """
    # Determine random numbers
    if distribution == 'uniform':
        r_1 = np.random.rand(pop.num_agents, pop.num_dimensions)
        r_2 = np.random.rand(pop.num_agents, pop.num_dimensions)
    elif distribution == 'gaussian':
        r_1 = np.random.randn(pop.num_agents, pop.num_dimensions)
        r_2 = np.random.randn(pop.num_agents, pop.num_dimensions)
    elif distribution == 'levy':
        r_1 = _random_levy(size=(pop.num_agents, pop.num_dimensions))
        r_2 = _random_levy(size=(pop.num_agents, pop.num_dimensions))
    else:
        raise OperatorsError('Invalid distribution!')

    # Choose the PSO version = 'inertial' or 'constriction'
    if version == 'inertial':
        # Find new velocities
        pop.velocities = factor * pop.velocities + r_1 * self_conf * (
                pop.particular_best_positions - pop.positions) + \
                         r_2 * swarm_conf * (np.tile(pop.global_best_position, (pop.num_agents, 1)) - pop.positions)
    elif version == 'constriction':
        # Find the constriction factor chi using phi
        phi = self_conf + swarm_conf
        if phi > 4:
            chi = 2 * factor / np.abs(2 - phi - np.sqrt(phi ** 2 - 4 * phi))
        else:
            chi = np.sqrt(factor)

        # Find new velocities
        pop.velocities = chi * (pop.velocities +
                                r_1 * self_conf * (pop.particular_best_positions - pop.positions) +
                                r_2 * swarm_conf * (np.tile(pop.global_best_position, (pop.num_agents, 1)) -
                                                    pop.positions))
    else:
        raise OperatorsError('Invalid swarm_dynamic version')

    # Move each agent using velocity's information
    pop.positions += pop.velocities


# %% INTERNAL METHODS

def _random_levy(size, beta=1.5):
    """
    This is an internal method to draw a random number (or array) using the Levy stable distribution via the
    Mantegna's algorithm.
        R. N. Mantegna and H. E. Stanley, “Stochastic Process with Ultraslow Convergence to a Gaussian: The Truncated
        Levy Flight,” Phys. Rev. Lett., vol. 73, no. 22, pp. 2946–2949, 1994.

    :param size: optional
        Size can be a tuple with all the dimensions. Behaviour similar to ``numpy.random.standard_normal``.
    :param float beta: optional.
        Levy distribution parameter. The default is 1.5.

    :return: numpy.array
    """
    # Calculate x's std dev (Mantegna's algorithm)
    sigma = ((np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2)) / (
            np.math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))) ** (1 / beta)

    # Determine x and y using normal distributions with sigma_y = 1
    x = sigma * np.random.standard_normal(size)
    y = np.abs(np.random.standard_normal(size))
    z = np.random.standard_normal(size)

    # Calculate the random number with levy stable distribution
    return z * x / (y ** (1 / beta))


def get_rotation_matrix(dimensions, angle=0.3927):
    """
    Determine the rotation matrix by multiplying all the rotation matrices for each combination of 2D planes.

    :param int dimensions:
        Number of dimensions. Only positive integers greater than one.
    :param float angle: optional.
        Rotation angle (in radians). The default is 0.3927 radians (or 22.5 degrees).

    :return: numpy.array
        Rotation matrix to use over the population positions.
    """
    # Initialise the rotation matrix
    rotation_matrix = np.eye(dimensions)

    # Find the combinations without repetitions
    planes = list(_get_combinations(range(dimensions), 2))

    # Create the rotation matrix
    for xy in range(len(planes)):
        # Read dimensions
        x, y = planes[xy]

        # (Re)-initialise a rotation matrix for each plane
        rotation_plane = np.eye(dimensions)

        # Assign corresponding values
        rotation_plane[[x, x, y, y], [x, y, x, y]] = [np.cos(angle), -np.sin(angle), np.sin(angle), np.cos(angle)]
        rotation_matrix = np.matmul(rotation_matrix, rotation_plane)

    return rotation_matrix


class OperatorsError(Exception):
    """
    Simple OperatorError to manage exceptions.
    """
    pass


# %% TOOLS TO HANDLE THE OPERATORS

def obtain_operators(num_vals=5):
    """
    Generate a list of all the available search operators with a given number of values for each parameter (if so).
    Each element of this list has the following structure:
            search_operator = ('name_of_the_operator',
                               {'parameter1': [value1, value2, ...],
                                'parameter2': [value2, value2, ...],
                                ... },
                               'name_of_the_selector')

    Available selectors from ``population.__selectors__`` are: 'all', 'greedy', 'metropolis', and 'probabilistic'.

    :param int num_vals: optional
        Number of values to generate per each numerical parameter in a search operator. The default is 5.

    :return: list
    """
    return [
        # First line for the initial search operator
        ('random_search', dict(scale=[1.0], distribution=['uniform']), ['greedy']),
        (
            'central_force_dynamic',
            dict(
                gravity=[*np.linspace(0.0, 0.01, num_vals)],
                alpha=[*np.linspace(0.0, 0.01, num_vals)],
                beta=[*np.linspace(1.00, 2.00, num_vals)],
                dt=[*np.linspace(0.0, 2.0, num_vals)]),
            __selectors__),
        # (
        #    'differential_crossover',
        #    dict(
        #        crossover_rate=[*np.linspace(0.0, 1.0, num_vals)],
        #        version=['binomial', 'exponential']),
        #    __selectors__),
        (
            'differential_mutation',
            dict(
                expression=['rand', 'best', 'current', 'current-to-best', 'rand-to-best', 'rand-to-best-and-current'],
                num_rands=[1, 2, 3],
                factor=[*np.linspace(0.0, 2.5, num_vals)]),
            __selectors__),
        (
            'firefly_dynamic',
            dict(
                distribution=['uniform', 'gaussian', 'levy'],
                alpha=[*np.linspace(0.0, 0.5, num_vals)],
                beta=[*np.linspace(0.01, 1.0, num_vals)],
                gamma=[*np.linspace(1.0, 1000.0, num_vals)]),
            __selectors__),
        (
            'genetic_crossover',
            dict(
                pairing=['rank', 'cost', 'random', 'tournament_2_100', 'tournament_2_75', 'tournament_2_50',
                         'tournament_3_100', 'tournament_3_75', 'tournament_3_50'],
                crossover=['single', 'two', 'uniform', 'blend', 'linear_0.5_0.5'],
                mating_pool_factor=[*np.linspace(0.1, 0.9, num_vals)]),
            __selectors__),
        (
            'genetic_mutation',
            dict(
                scale=[*np.linspace(0.01, 1.0, num_vals)],
                elite_rate=[*np.linspace(0.0, 0.9, num_vals)],
                mutation_rate=[*np.linspace(0.1, 0.9, num_vals)],
                distribution=['uniform', 'gaussian', 'levy']),
            __selectors__),
        (
            'gravitational_search',
            dict(
                gravity=[*np.linspace(0.0, 1.0, num_vals)],
                alpha=[*np.linspace(0.0, 0.04, num_vals)]),
            __selectors__),
        (
            'random_flight',  # Particular case for Levy flight
            dict(
                scale=[*np.linspace(0.01, 1.0, num_vals)],
                distribution=['levy'],
                beta=[*np.linspace(1.00, 2.00, num_vals)]),
            __selectors__),
        (
            'random_flight',
            dict(
                scale=[*np.linspace(0.01, 1.0, num_vals)],
                distribution=['uniform', 'gaussian']),
            __selectors__),
        (
            'local_random_walk',
            dict(
                probability=[*np.linspace(0.01, 0.99, num_vals)],
                scale=[*np.linspace(0.01, 1.0, num_vals)],
                distribution=['uniform', 'gaussian', 'levy']),
            __selectors__),
        (
            'random_sample',
            dict(),
            __selectors__),
        (
            'random_search',
            dict(
                scale=[*np.linspace(0.01, 1.0, num_vals)],
                distribution=['uniform', 'gaussian', 'levy']),
            __selectors__),
        (
            'spiral_dynamic',
            dict(
                radius=[*np.linspace(0.001, 0.999, num_vals)],
                angle=[*np.linspace(0.0, 180.0, num_vals)],
                sigma=[*np.linspace(0.0, 0.5, num_vals)]),
            __selectors__),
        (
            'swarm_dynamic',
            dict(
                factor=[*np.linspace(0.01, 1.0, num_vals)],
                self_conf=[*np.linspace(0.01, 4.99, num_vals)],
                swarm_conf=[*np.linspace(0.01, 4.99, num_vals)],
                version=['inertial', 'constriction'],
                distribution=['uniform', 'gaussian', 'levy']),
            __selectors__)
    ]


def build_operators(heuristics=obtain_operators(), file_name='operators_collection'):
    """
    Create a text file containing a list of all the available search operators, with the same structure as that
    generated by ``operators.obtain_operators``.

    :param list heuristics: optional.
        A list of available search operators. The default is ``obtain_operators()``.
    :param str file_name: optional.
        Customise the file name. The default is 'operators_collection'.

    :return: None.

    """
    # Counters: [classes, methods]
    total_counters = [0, 0]

    # Check if collections exists
    if not os.path.isdir('collections'):
        os.mkdir('collections')

    # Initialise the collection of simple heuristics
    if file_name[-4:] == '.txt':
        file_name = file_name[:-4]
    file = open('collections/' + file_name + '.txt', 'w')

    # For each simple heuristic, read their parameters and values
    for operator, parameters, selectors in heuristics:
        # Update the total classes counter
        total_counters[0] += 1

        # Read the number of parameters and how many values have each one
        num_parameters = len(parameters)
        num_selectors = len(selectors)
        num_combinations: int = 0
        if num_parameters > 0:
            # Read the name and possible values of parameters
            par_names = list(parameters.keys())
            par_values = list(parameters.values())

            # Find the number of values for each parameter
            par_num_values = [np.size(x) for x in par_values]

            # Determine the number of combinations
            num_combinations = int(np.prod(par_num_values))

            # Create the table of all possible combinations (index/parameter)
            indices = [x.flatten() for x in np.meshgrid(*list(map(lambda y: np.arange(y), par_num_values)))]

            # For each combination, create a single dictionary which
            # corresponds to a simple search operator
            for combi in range(num_combinations):
                list_tuples = [
                    (par_names[k], par_values[k][indices[k][combi]])
                    for k in range(num_parameters)]
                simple_par_combination = dict(list_tuples)
                for selector in selectors:
                    file.write("('{}', {}, '{}')\n".format(
                        operator, simple_par_combination, selector))
        elif num_parameters == 0:
            num_combinations = num_selectors
            for selector in selectors:
                file.write("('{}', {}, '{}')\n".format(
                    operator, '{}', selector))

        # Update the total combination counter
        total_counters[1] += num_combinations * num_selectors

        print(f"{operator}: parameters={num_parameters}, " + f"combinations:{num_combinations}")

    # Close the file and print how many types and specific operators were stored
    file.close()
    print("-" * 50 + "--\nTOTAL: families=%d, operators=%d" % tuple(total_counters))


def process_operators(simple_heuristics):
    """
    Decode the list of search operator and deliver two lists, one with the ready-to-execute strings of these operators
    and another with strings of their associated selectors.

    :param list simple_heuristics:
        A list of all the search operators to use. It may look like that saved using ``operators.build_operators``.

    :returns: (list, list)
        out[0] - executable_operators is a list of ready-to-execute string of search operators, and
        out[1] - selectors is the list of strings of the selectors associated to operators.
    """
    # Initialise the list of callable operators (simple heuristics)
    executable_operators = []
    selectors = []

    # For each simple heuristic, read their parameters and values
    for operator, parameters, selector in simple_heuristics:
        # Store selectors
        selectors.append(selector)

        if len(parameters) >= 0:
            sep = ','
            str_parameters = []

            for parameter, value in parameters.items():

                # Check if a value is string
                if type(value) == str:
                    str_parameters.append("{}='{}'".format(parameter, value))
                else:
                    str_parameters.append('{}={}'.format(parameter, value))

            # Create an executable string with given arguments
            full_string = '{}({})'.format(operator, sep.join(str_parameters))
        else:
            # Create an executable string with default arguments
            full_string = '{}()'.format(operator)

        # Store the read operator
        executable_operators.append(full_string)

    # Return two lists of executable operators and selectors
    return executable_operators, selectors


# %% AUTOMATIC RUN

if __name__ == '__main__':
    """
    Automatically create a collection of search operators using ```operators.obtain_operators(num_vals=5)``` and
    save it as 'automatic.txt'.
    """
    build_operators(obtain_operators(num_vals=5), file_name='automatic')
