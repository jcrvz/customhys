# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 14:54:31 2020

@author: Jorge Mario Cruz-Duarte (jcrvz.github.io)
"""

import numpy as np
from itertools import combinations as _get_combinations

__all__ = ['local_random_walk', 'random_search', 'random_sample',
           'rayleigh_flight', 'levy_flight', 'differential_mutation',
           'differential_crossover', 'firefly_dynamic', 'swarm_dynamic',
           'gravitational_search', 'central_force_dynamic', 'spiral_dynamic',
           'genetic_mutation', 'genetic_crossover']

# -------------------------------------------------------------------------
#    PERTURBATORS
# -------------------------------------------------------------------------
def random_sample(pop):
    """
    Performs a random sampling using a uniform distribution in [-1, 1].

    Parameters
    ----------
    pop : population
        It is a population object.

    Returns
    -------
    None.

    """
    # Create random positions using random numbers between -1 and 1
    pop.positions = np.random.uniform(
        -1, 1, (pop.num_agents, pop.num_dimensions))

    # Check constraints
    if pop.is_constrained:
        pop._check_simple_constraints()


def random_search(pop, scale=0.01):
    """
    Performs a random walk using a uniform distribution in [-1, 1].

    Parameters
    ----------
    pop : population
        It is a population object.
    scale : float, optional
        It is the step scale between [0.0, 1.0]. The default is 0.01.

    Returns
    -------
    None.

    """
    # Check the scale value
    _check_parameter('scale')

    # Move each agent using uniform random displacements
    pop.positions += scale * \
        np.random.uniform(-1, 1, (pop.num_agents, pop.num_dimensions))

    # Check constraints
    if pop.is_constrained:
        pop._check_simple_constraints()


def rayleigh_flight(pop, scale=0.01):
    """
    Perform a Rayleigh flight using a normal standard distribution.

    Parameters
    ----------
    pop : population
        It is a population object.
    scale : float, optional
        It is the step scale between [0.0, 1.0]. The default is 0.01.

    Returns
    -------
    None.

    """
    # Check the scale value
    _check_parameter('scale')

    # Move each agent using gaussian random displacements
    pop.positions += scale * \
        np.random.standard_normal((pop.num_agents, pop.num_dimensions))

    # Check constraints
    if pop.is_constrained:
        pop._check_simple_constraints()


def levy_flight(pop, scale=1.0, beta=1.5):
    """
    Perform a Lévy flight by using the Mantegna's algorithm.

    Parameters
    ----------
    pop : population
        It is a population object.
    scale : float, optional
        It is the step scale between [0.0, 1.0]. The default is 1.0.
    beta : float, optional
        It is the distribution parameter between [1.0, 3.0]. The default
        is 1.5.

    Returns
    -------
    None.

    """
    # Check the scale and beta value
    _check_parameter('scale')
    _check_parameter('beta', (1.0, 3.0))

    # Calculate x's std dev (Mantegna's algorithm)
    sigma = ((np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2)) /
             (np.math.gamma((1 + beta) / 2) * beta *
              (2 ** ((beta - 1) / 2)))) ** (1 / beta)

    # Determine x and y using normal distributions with sigma_y = 1
    x = sigma * np.random.standard_normal((pop.num_agents,
                                           pop.num_dimensions))
    y = np.abs(np.random.standard_normal((pop.num_agents,
                                          pop.num_dimensions)))

    # Calculate the random number with levy stable distribution
    levy_random = x / (y ** (1 / beta))

    # Determine z as an additional normal random number
    z = np.random.standard_normal((pop.num_agents, pop.num_dimensions))

    # Move each agent using levy random displacements
    pop.positions += scale * z * levy_random * \
        (pop.positions - np.tile(pop.global_best_position,
                                 (pop.num_agents, 1)))

    # Check constraints
    if pop.is_constrained:
        pop._check_simple_constraints()


def swarm_dynamic(pop, factor=1.0, self_conf=2.4, swarm_conf=2.6,
                  version="constriction"):
    """
    Performs a swarm movement by using the inertial or constriction
    dynamics from Particle Swarm Optimisation (PSO).

    Parameters
    ----------
    pop : population
        It is a population object.
    factor : float, optional
        Inertial or Kappa factor, depending of which PSO version is set.
        The default is 1.0.
    self_conf : float, optional
        Self confidence factor. The default is 2.4.
    swarm_conf : float, optional
        Swarm confidence factor. The default is 2.6.
    version : str, optional
        Version of the Particle Swarm Optimisation strategy. Currently, it
        can be 'constriction' or 'inertial'. The default is "constriction".

    Returns
    -------
    None.

    """
    # Check the scale and beta value
    _check_parameter('factor')
    _check_parameter('self_conf', (0.0, 10.0))
    _check_parameter('swarm_conf', (0.0, 10.0))

    # Determine random numbers
    r_1 = self_conf * np.random.rand(pop.num_agents, pop.num_dimensions)
    r_2 = swarm_conf * np.random.rand(pop.num_agents, pop.num_dimensions)

    # Choose the PSO version = 'inertial' or 'constriction'
    if version == "intertial":
        # Find new velocities
        pop.velocities = factor * pop.velocities + r_1 * (
            pop.particular_best_positions - pop.positions) + \
            r_2 * (np.tile(pop.global_best_position, (pop.num_agents, 1)) -
                   pop.positions)
    elif version == "constriction":
        # Find the constriction factor chi using phi
        phi = self_conf + swarm_conf
        if phi > 4:
            chi = 2 * factor / np.abs(2 - phi - np.sqrt(phi ** 2 - 4 * phi))
        else:
            chi = np.sqrt(factor)
    else:
        OperatorsError('Invalid swarm_dynamic version')

        # Find new velocities
        pop.velocities = chi * (pop.velocities + r_1 * (
            pop.particular_best_positions - pop.positions) +
            r_2 * (np.tile(pop.global_best_position, (pop.num_agents, 1)) -
                   pop.positions))

    # Move each agent using velocity's information
    pop.positions += pop.velocities

    # Check constraints
    if pop.is_constrained:
        pop._check_simple_constraints()


# Before: mutation_de
def differential_mutation(pop, expression="current-to-best", num_rands=1,
                          factor=1.0):
    """
    Mutates the population positions using Differential Evolution (DE)

    Parameters
    ----------
    pop : population
        It is a population object.
    expression : str, optional
        Type of DE mutation. Available mutations: "rand", "best",
        "current", "current-to-best", "rand-to-best",
        "rand-to-bestandcurrent". The default is "current-to-best".

    num_rands : TYPE, optional
        DESCRIPTION. The default is 1.
    factor : TYPE, optional
        DESCRIPTION. The default is 1.0.

    Returns
    -------
    None.

    """
    # Check the scale and beta value
    _check_parameter('num_rands', (1, 10), int)
    _check_parameter('factor', (0.0, 2.0))

    # Create mutants using the expression provided in scheme
    if expression == "rand":
        mutant = pop.positions[np.random.permutation(pop.num_agents), :]

    elif expression == "best":
        mutant = np.tile(pop.global_best_position, (pop.num_agents, 1))

    elif expression == "current":
        mutant = pop.positions

    elif expression == "current-to-best":
        mutant = pop.positions + factor * \
            (np.tile(pop.global_best_position,
                     (pop.num_agents, 1)) -
             pop.positions[np.random.permutation(pop.num_agents), :])

    elif expression == "rand-to-best":
        mutant = pop.positions[np.random.permutation(pop.num_agents), :] + \
            factor * (np.tile(pop.global_best_position, (pop.num_agents, 1)) -
                      pop.positions[np.random.permutation(pop.num_agents), :])

    elif expression == "rand-to-best-and-current":
        mutant = pop.positions[np.random.permutation(
            pop.num_agents), :] + factor * (np.tile(
                pop.global_best_position, (pop.num_agents, 1)) -
                pop.positions[np.random.permutation(
                    pop.num_agents), :] + pop.positions[
                        np.random.permutation(
                            pop.num_agents), :] - pop.positions)
    else:
        mutant = []
        raise pop.__PopulationError('Invalid DE mutation scheme!')

    # Add random parts according to num_rands
    if num_rands >= 0:
        for _ in range(num_rands):
            mutant += factor * (pop.positions[np.random.permutation(
                pop.num_agents), :] - pop.positions[
                    np.random.permutation(pop.num_agents), :])
    else:
        raise pop.__PopulationError('Invalid DE mutation scheme!')

    # Replace mutant population in the current one
    pop.positions = mutant

    # Check constraints
    if pop.is_constrained:
        pop._check_simple_constraints()


def differential_crossover(pop, crossover_rate=0.5, version="binomial"):
    """
    Performs either the binomial or exponential crossover procedure from
    Differential Evolution (DE).

    Parameters
    ----------
    pop : population
        It is a population object.
    crossover_rate : float, optional
        Probability factor to perform the crossover. The default is 0.5.
    version : str, optional
        Crossover version. It can be 'binomial' or 'exponential'.
        The default is "binomial".

    Returns
    -------
    None.

    """
    # Check the scale and beta value
    _check_parameter('crossover_rate')

    # Binomial versio
    if version == "binomial":
        # Define indices
        indices = np.tile(np.arange(pop.num_dimensions), (pop.num_agents, 1))

        # Permute indices per dimension
        rand_indices = np.vectorize(np.random.permutation,
                                    signature='(n)->(n)')(indices)

        # Calculate the NOT condition (because positions already updated!)
        condition = np.logical_not((indices == rand_indices) | (
            np.random.rand(pop.num_agents, pop.num_dimensions) <=
            crossover_rate))

        # Reverse the ones to their previous positions
        pop.positions[condition] = pop.previous_positions[condition]
    #
    # Exponential version
    elif version == "exponential":
        # Perform the exponential crossover procedure
        for agent in range(pop.num_agents):
            for dim in range(pop.num_dimensions):
                # Initialise L and choose a random index n
                exp_var = 0
                n = np.random.randint(pop.num_dimensions)
                while True:
                    # Increase L and check the exponential crossover condition
                    exp_var += 1
                    if np.logical_not((np.random.rand() < crossover_rate) and
                                      (exp_var < pop.num_dimensions)):
                        break

                # Perform the crossover if the following condition is met
                if dim not in [(n + x) % pop.num_dimensions for x in
                               range(exp_var)]:
                    pop.positions[agent, dim] = pop.previous_positions[
                        agent, dim]
    #
    # Invalid version
    else:
        OperatorsError('Invalid differential_crossover version')

    # Check constraints
    if pop.is_constrained:
        pop._check_simple_constraints()


def local_random_walk(pop, probability=0.75, scale=1.0):
    """
    Performs the local random walk from Cuckoo Search (CS)

    Parameters
    ----------
    pop : population
        It is a population object.
    probability : float, optional
        It is the probability of discovering an alien egg (change an
        agent's position). The default is 0.75.
    scale : float, optional
        It is the step scale between [0.0, 1.0]. The default is 1.0.

    Returns
    -------
    None.

    """
    # Check the scale and beta value
    _check_parameter('probability')
    _check_parameter('scale')

    # Determine random numbers
    r_1 = np.random.rand(pop.num_agents, pop.num_dimensions)
    r_2 = np.random.rand(pop.num_agents, pop.num_dimensions)

    # Move positions with a displacement due permutations and probabilities
    pop.positions += scale * r_1 * (pop.positions[
        np.random.permutation(pop.num_agents), :] - pop.positions[
            np.random.permutation(pop.num_agents), :]) * np.heaviside(
                r_2 - probability, 0.0)

    # Check constraints
    if pop.is_constrained:
        pop._check_simple_constraints()


def spiral_dynamic(pop, radius=0.9, angle=22.5, sigma=0.1):
    """
    Performs the deterministic or stochastic spiral dynamic movement

    Parameters
    ----------
    pop : population
        It is a population object.
    radius : float, optional
        It is the convergence rate. The default is 0.9.
    angle : float, optional
        Rotation angle (in degrees). The default is 22.5 (degrees).
    sigma : float, optional
        Variation of random radii. The default is 0.1.
        Note: sigma equals 0.0 corresponds to the Deterministic Spiral.

    Returns
    -------
    None.

    """
    # Check the scale and beta value
    _check_parameter('radius')
    _check_parameter('angle', (0.0, 360.0))
    _check_parameter('sigma')

    # Determine the rotation matrix
    rotation_matrix = _get_rotation_matrix(pop.num_dimensions,
                                           np.deg2rad(angle))

    for agent in range(pop.num_agents):
        random_radii = np.random.uniform(radius - sigma, radius + sigma,
                                         pop.num_dimensions)
        # If random radii need to be constrained to [0, 1]:
        pop.positions[agent, :] = pop.global_best_position + random_radii * \
            np.matmul(rotation_matrix, (
                pop.positions[agent, :] - pop.global_best_position))

    # Check constraints
    if pop.is_constrained:
        pop._check_simple_constraints()


# Before: firefly
def firefly_dynamic(pop, epsilon="uniform", alpha=0.8, beta=1.0, gamma=1.0):
    """
    Performs movements accordint to the Firefly algorithm (FA)

    Parameters
    ----------
    pop : population
        It is a population object.
    epsilon : str, optional
        Type of random number. Possible options: 'gaussian', 'uniform'.
        The default is "uniform".
    alpha : TYPE, optional
        DESCRIPTION. The default is 0.8.
    beta : TYPE, optional
        DESCRIPTION. The default is 1.0.
    gamma : TYPE, optional
        DESCRIPTION. The default is 1.0.

    Returns
    -------
    None.

    """
    # Check the alpha, beta, and gamma value
    _check_parameter('alpha')
    _check_parameter('beta')
    _check_parameter('gamma', (0.0, 100.0))

    # Determine epsilon values
    if epsilon == "gaussian":
        epsilon_value = np.random.standard_normal(
            (pop.num_agents, pop.num_dimensions))

    elif epsilon == "uniform":
        epsilon_value = np.random.uniform(
            -0.5, 0.5, (pop.num_agents, pop.num_dimensions))
    else:
        epsilon_value = []
        raise pop.__PopulationError(
            "Epsilon is not valid: 'uniform' or 'gaussian'")

    # Initialise delta or difference between two positions
    difference_positions = np.zeros((pop.num_agents, pop.num_dimensions))

    for agent in range(pop.num_agents):
        # Select indices in order to avoid division by zero
        indices = (np.arange(pop.num_agents) != agent)

        # Determine all vectorial distances with respect to agent
        delta = pop.positions[indices, :] - np.tile(
            pop.positions[agent, :], (pop.num_agents - 1, 1))

        # Determine differences between lights
        delta_lights = np.tile(
            (pop.fitness[indices] - np.tile(
                pop.fitness[agent], (1, pop.num_agents-1))).transpose(),
            (1, pop.num_dimensions))

        # Find the total attraction for each agent
        difference_positions[agent, :] = np.sum(
            np.heaviside(-delta_lights, 0.0) * delta *
            np.exp(-gamma * np.linalg.norm(delta, 2, 1) ** 2), 0)

    # Move fireflies according to their attractions
    pop.positions += alpha * epsilon_value + beta * difference_positions

    # Check constraints
    if pop.is_constrained:
        pop._check_simple_constraints()


# Before: central_force
def central_force_dynamic(pop, gravity=1.0, alpha=0.5, beta=1.5, dt=1.0):
    """
    Central Force Optimisation (CFO)

    Parameters
    ----------
    pop : population
        It is a population object.
    gravity : float, optional
        It is the gravitational constant. The default is 1.0.
    alpha : float, optional
        It is the power mass factor. The default is 0.5.
    beta : float, optional
        It is the power distance factor. The default is 1.5.
    dt : float, optional
        It is the time interval between steps. The default is 1.0.

    Returns
    -------
    None.

    """
    # Check the gravity, alpha, beta, and dt value
    _check_parameter('alpha')
    _check_parameter('beta', (1.0, 3.0))
    _check_parameter('dt', (0.0, 10.0))
    _check_parameter('gravity')

    # Initialise acceleration
    acceleration = np.zeros((pop.num_agents, pop.num_dimensions))

    for agent in range(pop.num_agents):
        # Select indices in order to avoid division by zero
        indices = (np.arange(pop.num_agents) != agent)

        # Determine all masses differences with respect to agent
        delta_masses = pop.fitness[indices] - np.tile(
            pop.fitness[agent], (1, pop.num_agents - 1))

        # Determine all vectorial distances with respect to agent
        delta_positions = pop.positions[indices, :] - np.tile(
            pop.positions[agent, :], (pop.num_agents - 1, 1))

        distances = np.linalg.norm(delta_positions, 2, 1)

        # Find the quotient part    ! -> - delta_masses (cz minimisation)
        quotient = np.heaviside(-delta_masses, 0.0) * (
            np.abs(delta_masses) ** alpha) / (distances ** beta)

        # Determine the acceleration for each agent
        acceleration[agent, :] = gravity * np.sum(
            delta_positions * np.tile(quotient.transpose(),
                                      (1, pop.num_dimensions)), 0)

    pop.positions += 0.5 * acceleration * (dt ** 2)

    # Check constraints
    if pop.is_constrained:
        pop._check_simple_constraints()


def gravitational_search(pop, gravity=1.0, alpha=0.5):
    """
    Gravitational Search Algorithm (GSA) simplified

    Parameters
    ----------
    pop : population
        It is a population object.
    gravity : float, optional
        It is the initial gravitational value. The default is 1.0.
    alpha : float, optional
        It is the gravitational damping ratio. The default is 0.5.

    Returns
    -------
    None.

    """
    # Check the gravity, alpha, and epsilon value
    _check_parameter('gravity')
    _check_parameter('alpha')
    _check_parameter('epsilon', (0.0, 0.1))

    # Initialise acceleration
    acceleration = np.zeros((pop.num_agents, pop.num_dimensions))

    # Determine the gravitational constant
    gravitation = gravity * np.exp(- alpha * pop.iteration)

    # Determine mass for each agent
    raw_masses = (pop.fitness - np.tile(
        pop.current_worst_fitness, (1, pop.num_agents)))
    masses = (raw_masses / np.sum(raw_masses)).reshape(pop.num_agents)

    for agent in range(pop.num_agents):
        # Select indices in order to avoid division by zero
        indices = (np.arange(pop.num_agents) != agent)

        # Determine all vectorial distances with respect to agent
        delta_positions = pop.positions[indices, :] - np.tile(
            pop.positions[agent, :], (pop.num_agents - 1, 1))

        quotient = masses[indices] / (
            np.linalg.norm(delta_positions, 2, 1) + 1e-23)

        # Force interaction
        force_interaction = gravitation * np.tile(
            quotient.reshape(pop.num_agents - 1, 1),
            (1, pop.num_dimensions)) * delta_positions

        # Acceleration
        acceleration[agent, :] = np.sum(np.random.rand(
            pop.num_agents - 1, pop.num_dimensions) *
            force_interaction, 0)

    # Update velocities
    pop.velocities = acceleration + np.random.rand(
        pop.num_agents, pop.num_dimensions) * pop.velocities

    # Update positions
    pop.positions += pop.velocities

    # Check constraints
    if pop.is_constrained:
        pop._check_simple_constraints()


# Before: ga_crossover
def genetic_crossover(pop, pairing="cost", crossover="single",
                      mating_pool_factor=0.1):
    """
    Crossover mechanism from Genetic Algorithm (GA)

    Parameters
    ----------
    pop : population
        It is a population object.
    pairing : str, optional
        It indicates which pairing scheme to employ. Pairing scheme
        available are: 'cost' (Roulette Wheel or Cost Weighting), 'rank'
        (Rank Weighting), 'tournament', 'random', and 'even-odd'.
        Tournament size (tp) and probability (tp) can be encoded such as
        'tournament_{ts}_{tp}', {ts} and {tp}. Writing only 'tournament' is
        similar to specify 'tournament_3_100'.
        The default is 'cost'.
    crossover : str, optional
        It indicates which crossover scheme to employ. Crossover scheme
        availale are: 'single', 'two', 'uniform', 'blend', and 'linear'.
        Likewise 'tournament' pairing, coefficients of 'linear' are enconded
        such as 'linear_{coeff1}_{coeff2}' where the offspring is determined
        as follows: offspring = coeff1 * father + coeff2 * mother.
        The default is 'single'.
    mating_pool_factor : float, optional
        It indicates the proportion of population to disregard.
        The default is 0.1.

    Returns
    -------
    None.

    """
    # Check the mating_pool_factor value
    _check_parameter('mating_pool_factor')

    # Mating pool size
    num_mates = int(np.round(mating_pool_factor * pop.num_agents))

    # Number of offsprings (or couples)
    num_couples = pop.num_agents - num_mates

    # Get the mating pool using the natural selection
    mating_pool_indices = np.argsort(pop.fitness)[:num_mates]

    # Get parents (at least a couple per offspring)
    if len(pairing) > 10:  # if pairing = 'tournament_2_100', for example
        pairing, tournament_size, tournament_probability = \
            pairing.split("_")
    else:  # dummy (it must not be used)
        tournament_size, tournament_probability = '3', '100'

    # Perform the pairing procedure
    tournament_size = int(tournament_size)
    #
    # Roulette Wheel (Cost Weighting) Selection
    if pairing == "cost":
        # Cost normalisation from mating pool: cost-min(cost @ non mates)
        normalised_cost = pop.fitness[mating_pool_indices] - np.min(
            pop.fitness[np.setdiff1d(
                np.arange(pop.num_agents), mating_pool_indices)])

        # Determine the related probabilities
        probabilities = np.abs(normalised_cost / np.sum(normalised_cost))

        # Perform the roulette wheel selection and return couples
        couple_indices_ = np.searchsorted(
            np.cumsum(probabilities), np.random.rand(2 * num_couples))

        # Return couples
        couple_indices = couple_indices_.reshape((2, -1))
    #
    # Roulette Wheel (Rank Weighting) Selection
    elif pairing == "rank":
        # Determine the probabilities
        probabilities = (mating_pool_indices.size - np.arange(
            mating_pool_indices.size)) / np.sum(
                np.arange(mating_pool_indices.size) + 1)

        # Perform the roulette wheel selection and return couples
        couple_indices_ = np.searchsorted(
            np.cumsum(probabilities), np.random.rand(2*num_couples))

        # Return couples
        couple_indices = couple_indices_.reshape((2, -1))
    #
    # Tournament pairing
    elif pairing == "tournament":
        # Calculate probabilities
        probability = float(tournament_probability)/100
        probabilities = probability * (
            (1 - probability) ** np.arange(tournament_size))

        # Initialise the mother and father indices
        couple_indices = np.full((2, num_couples), np.nan)

        # Perform tournaments until all mates are selected
        for couple in range(num_couples):
            mate = 0
            while mate < 2:
                # Choose tournament candidates
                random_indices = mating_pool_indices[
                    np.random.permutation(mating_pool_indices.size)[
                        :tournament_size]]

                # Determine the candidate fitness values
                candidates_indices = random_indices[
                    np.argsort(pop.fitness[random_indices])]

                # Find the best according to its fitness and probability
                winner = candidates_indices[
                    np.random.rand(tournament_size) < probabilities]
                if winner.size > 0:
                    couple_indices[mate, couple] = int(winner[0])
                    mate += 1
    #
    # Random pairing
    elif pairing == "random":
        # Return two random indices from mating pool
        couple_indices = mating_pool_indices[np.random.randint(
            mating_pool_indices.size, size=(2, num_couples))]
    #
    # Even-and-Odd pairing
    elif pairing == "even-odd":
        # Check if the num of mates is even
        mating_pool_size = mating_pool_indices.size - \
            (mating_pool_indices.size % 2)
        half_size = mating_pool_size // 2

        # Dummy indices according to the mating pool size
        remaining = num_couples - half_size
        if remaining > 0:
            dummy_indices = np.tile(
                np.reshape(np.arange(mating_pool_size),
                           (-1, 2)).transpose(),
                (1, int(np.ceil(num_couples / half_size))))
        else:
            dummy_indices = np.reshape(np.arange(mating_pool_size),
                                       (-1, 2)).transpose()

        # Return couple_indices
        couple_indices = mating_pool_indices[
            dummy_indices[:, :num_couples]]
    #
    # No pairing procedure recognised
    else:
        pop.__PopulationError("Invalid pairing method")

    # Identify offspring indices
    offspring_indices = np.setdiff1d(
        np.arange(pop.num_agents), mating_pool_indices, True)

    # Prepare crossover variables
    if len(crossover) > 7:  # if crossover = 'linear_0.5_0.5', for example
        crossover, coeff1, coeff2 = crossover.split("_")
        coefficients = [float(coeff1), float(coeff2)]
    else:  # dummy (it must not be used)
        coefficients = [np.nan, np.nan]

    # Perform crossover and assign to population
    parent_indices = couple_indices.astype(np.int64)
    #
    # Single-Point Crossover
    if crossover == "single":
        # Determine the single point per each couple
        single_points = np.tile(np.random.randint(
            pop.num_dimensions, size=parent_indices.shape[1]),
            (pop.num_dimensions, 1)).transpose()

        # Crossover condition mask
        crossover_mask = np.tile(
            np.arange(pop.num_dimensions), (parent_indices.shape[1], 1)
            ) <= single_points

        # Get father and mother
        father_position = pop.positions[parent_indices[0, :], :]
        mother_position = pop.positions[parent_indices[1, :], :]

        # Initialise offsprings with mother positions
        offsprings = mother_position
        offsprings[crossover_mask] = father_position[crossover_mask]
    #
    # Two-Point Crossover
    # OPTIMIZE It can be extended to multiple points
    elif crossover == "two":
        # Find raw points
        raw_points = np.sort(np.random.randint(
            pop.num_dimensions, size=(parent_indices.shape[1], 2)))

        # Determine the single point per each couple
        points = [np.tile(raw_points[:, x], (
            pop.num_dimensions, 1)).transpose() for x in
            range(raw_points.shape[1])]

        # Range matrix
        dummy_matrix = np.tile(np.arange(pop.num_dimensions),
                               (parent_indices.shape[1], 1))

        # Crossover condition mask (only for two points)
        crossover_mask = ((dummy_matrix <= points[0]) |
                          (dummy_matrix > points[1]))
        # OPTIMIZE : crossover_mask = points[1] < dummy_matrix <= points[0]

        # Get father and mother
        father_position = pop.positions[parent_indices[0, :], :]
        mother_position = pop.positions[parent_indices[1, :], :]

        # Initialise offsprings with mother positions
        offsprings = mother_position
        offsprings[crossover_mask] = father_position[crossover_mask]
    #
    # Uniform Crossover
    elif crossover == "uniform":
        # Crossover condition mask (only for two points)
        crossover_mask = np.random.rand(
            parent_indices.shape[1], pop.num_dimensions) < 0.5

        # Get father and mother
        father_position = pop.positions[parent_indices[0, :], :]
        mother_position = pop.positions[parent_indices[1, :], :]

        # Initialise offsprings with mother positions
        offsprings = mother_position
        offsprings[crossover_mask] = father_position[crossover_mask]
    #
    # Random blending crossover
    elif crossover == "blend":
        # Initialise random numbers between 0 and 1
        beta_values = np.random.rand(
            parent_indices.shape[1], pop.num_dimensions)

        # Get father and mother
        father_position = pop.positions[parent_indices[0, :], :]
        mother_position = pop.positions[parent_indices[1, :], :]

        # Determine offsprings with father and mother positions
        offsprings = beta_values * father_position +\
            (1 - beta_values) * mother_position
    #
    # Linear Crossover: offspring = coeff[0] * father + coeff[1] * mother
    elif crossover == "linear":
        # Get father and mother
        father_position = pop.positions[parent_indices[0, :], :]
        mother_position = pop.positions[parent_indices[1, :], :]

        # Determine offsprings with father and mother positions
        offsprings = coefficients[0] * father_position +\
            coefficients[1] * mother_position
    #
    # No crossover method recognised
    else:
        pop.__PopulationError("Invalid pairing method")

    # Store offspring positions in the current population
    pop.positions[offspring_indices, :] = offsprings


# Before: ga_mutation
def genetic_mutation(pop, elite_rate=0.0, mutation_rate=0.2,
                     distribution="uniform", sigma=1.0):
    """
    Mutation mechanism from Genetic Algorithm (GA)

    Parameters
    ----------
    pop : population
        It is a population object.
    elite_rate : float, optional
        It is the proportion of population to preserve.
        The default is 0.0 (no elite agent).
    mutation_rate : float, optional
        It is the proportion of population to mutate.
        The default is 0.2.
    distribution : str, optional
        It indicates the random distribution that power the mutation.
        There are only two distribution available: 'uniform' and 'gaussian'.
        The default is 'uniform'.
    sigma : float, optional
        It is the scale factor of the mutations. The default is 1.0.

    Returns
    -------
    None.

    """
    # Check the elite_rate, mutation_rate, and sigma value
    _check_parameter('elite_rate')
    _check_parameter('mutation_rate')
    _check_parameter('sigma')

    # Calculate the number of elite agents
    num_elite = int(np.ceil(pop.num_agents * elite_rate))

    # If num_elite equals num_agents then do nothing, or ...
    if (num_elite < pop.num_agents):
        # Number of mutations to perform
        num_mutations = int(np.round(pop.num_agents * pop.num_dimensions
                                     * mutation_rate))

        # Identify mutable agents
        dimension_indices = np.random.randint(
            0, pop.num_dimensions, num_mutations)

        if num_elite > 0:
            agent_indices = np.argsort(pop.fitness)[
                np.random.randint(num_elite, pop.num_agents,
                                  num_mutations)]
        else:
            agent_indices = np.random.randint(num_elite, pop.num_agents,
                                              num_mutations)

        # Transform indices
        rows, columns = np.meshgrid(agent_indices, dimension_indices)

        # Perform mutation according to the random distribution
        if distribution == "uniform":
            mutants = sigma * np.random.uniform(-1, 1, num_mutations ** 2)

        elif distribution == "gaussian":
            # Normal with mu = 0 and sigma = parameter
            mutants = sigma * np.random.standard_normal(num_mutations ** 2)

        # Store mutants
        pop.positions[rows.flatten(), columns.flatten()] = mutants


def _get_rotation_matrix(dimensions, angle=0.39269908169872414):
    """
    Determine the rotation matrix by multiplying all the rotation matrices for
    each combination of 2D planes.

    Parameters
    ----------
    dimensions : int
        Number of dimensions. Only positive integers greater than one.
    angle : float, optional
        Rotation angle. Only positive values.
        The default is 0.39269908169872414.

    Returns
    -------
    rotation_matrix : ndarray
        Rotation matrix to use over the population positions.

    """
    # Check the dimensions and angle value
    _check_parameter('dimensions', (2, np.inf), int)
    _check_parameter('angle', (0.0, 2*np.pi))

    # Initialise the rotation matrix
    rotation_matrix = np.eye(dimensions)

    # Find the combinations without repetions
    planes = list(_get_combinations(range(dimensions), 2))

    # Create the rotation matrix
    for xy in range(len(planes)):
        # Read dimensions
        x, y = planes[xy]

        # (Re)-initialise a rotation matrix for each plane
        rotation_plane = np.eye(dimensions)

        # Assign corresponding values
        rotation_plane[x, y] = np.cos(angle)
        rotation_plane[y, y] = np.cos(angle)
        rotation_plane[x, y] = -np.sin(angle)
        rotation_plane[y, x] = np.sin(angle)

        rotation_matrix = np.matmul(rotation_matrix, rotation_plane)

    return rotation_matrix


def _check_parameter(parameter, interval=(0.0, 1.0),
                     par_type=float):
    """
    Check if a parameter or variable is into an interval.

    Parameters
    ----------
    parameter: str
        Variable's name to check.
    interval : tuple, optional
        A tuple corresponding to the interval. The default is (0.0, 1.0).
    par_type : type, optional
        The parameter's type is also checked. The default is float.

    Raises
    ------
    PopulationError if the parameter does not pass the check.

    Returns
    -------
    None.

    """
    # Prepare some variables to perfom the checking
    assert isinstance(parameter, str)
    par_value = eval(parameter)
    assert isinstance(par_value, par_type)

    # Check if the parameter value is into the interval
    if not interval[0] <= par_value <= interval[1]:
        raise OperatorsError(
            "Invalid value! {} = {} is not in [{}, {}]".format(
                parameter, par_value, interval[0], interval[1]))


class OperatorsError(Exception):
    """
    Simple OperatorError to manage exceptions.
    """
    pass


# ---------------------------------------------------------------------------
# GENERATOR OF SEARCH OPERATORS
# ---------------------------------------------------------------------------

def _obtain_operators(num_vals=5):
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
                probability=np.linspace(0.0, 1.0, num_vals),
                scale=np.linspace(0.0, 1.0, num_vals)),
            "greedy"),
        (
            "random_search",
            dict(
                scale=np.linspace(0.0, 1.0, num_vals)),
            "greedy"),
        (
            "random_sample",
            dict(),
            "greedy"),
        (
            "rayleigh_flight",
            dict(
                scale=np.linspace(0.0, 1.0, num_vals)),
            "greedy"),
        (
            "levy_flight",
            dict(
                scale=np.linspace(0.0, 1.0, num_vals),
                beta=[1.5]),
            "greedy"),
        (
            "differential_mutation",
            dict(
                expression=["rand", "best", "current", "current-to-best",
                            "rand-to-best", "rand-to-best-and-current"],
                num_rands=[1, 2, 3],
                factor=np.linspace(0.0, 2.0, num_vals)),
            "greedy"),
        (
            'differential_crossover',
            dict(
                crossover_rate=np.linspace(0.0, 1.0, num_vals),
                version=["binomial", "exponential"]),
            "greedy"),
        (
            "firefly_dynamic",
            dict(
                epsilon=["uniform", "gaussian"],
                alpha=np.linspace(0.0, 1.0, num_vals),
                beta=[1.0],
                gamma=np.linspace(1.0, 100.0, num_vals)),
            "greedy"),
        (
            "swarm_dynamic",
            dict(
                factor=np.linspace(0.0, 1.0, num_vals),
                self_conf=np.linspace(0.0, 5.0, num_vals),
                swarm_conf=np.linspace(0.0, 5.0, num_vals),
                version=["inertial", "constriction"]),
            "all"),
        (
            "gravitational_search",
            dict(
                gravity=np.linspace(0.0, 1.0, num_vals),
                alpha=np.linspace(0.0, 1.0, num_vals),
                epsilon=[1e-23]),
            "all"),
        (
            "central_force_dynamic",
            dict(
                gravity=np.linspace(0.0, 1.0, num_vals),
                alpha=np.linspace(0.0, 1.0, num_vals),
                beta=np.linspace(0.0, 3.0, num_vals),
                dt=[1.0]),
            "all"),
        (
            "spiral_dynamic",
            dict(
                radius=np.linspace(0.0, 1.0, num_vals),
                angle=np.linspace(0.0, 180.0, num_vals),
                sigma=np.linspace(0.0, 0.5, num_vals)),
            "all"),
        (
            "genetic_mutation",
            dict(
                elite_rate=np.linspace(0.0, 1.0, num_vals),
                mutation_rate=np.linspace(0.0, 1.0, num_vals),
                distribution=["uniform", "gaussian"],
                sigma=np.linspace(0.0, 1.0, num_vals)),
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
                mating_pool_factor=np.linspace(0.0, 1.0, num_vals)),
            "all")
        ]


def _build_operators(heuristics=_obtain_operators(),
                     file_name="operators_collection"):
    """
    Create a text file containing all possible combinations of parameter
    values for each search operator.

    Parameters
    ----------
    heuristics : list, optional
        A list of available search operators. The default is
        _get_search_operators().
    file_name : str, optional
        Customise the file name. The default is 'operators_collection'

    Returns
    -------
    None.

    """
    # Counters: [classes, methods]
    total_counters = [0, 0]

    # Initialise the collection of simple heuristics
    file = open(file_name + '.txt', 'w')

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
            par_num_values = [np.np.size(x) for x in par_values]

            # Determine the number of combinations
            num_combinations = np.prod(par_num_values)

            # Create the table of all possible combinations (index/parameter)
            indices = [x.flatten() for x in np.meshgrid(
                *list(map(lambda x: np.arange(x), par_num_values)))]

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


def _process_operators(self, simple_heuristics):
    """
    Decode the list of operators or heuristics and deliver two lists, one
    with ready-to-execute strings of operators and another with strings of
    their associated selectors.

    Parameters
    ----------
    simple_heuristics : list
        A list of all the search operators to use.

    Returns
    -------
    executable_operators : list
        A list of ready-to-execute string of search operators.
    selectors : list
        A list of strings of the selectors associated to operators.
    """
    # Initialise the list of callable operators (simple heuristics)
    executable_operators = []
    selectors = []

    # For each simple heuristic, read their parameters and values
    for operator, parameters, selector in simple_heuristics:
        # Store selectors
        selectors.append(selector)

        if len(parameters) >= 0:
            sep = ","
            str_parameters = []

            for parameter, value in parameters.items():

                # Check if a value is string
                if type(value) == str:
                    str_parameters.append("{}='{}'".format(
                        parameter, value))
                else:
                    str_parameters.append("{}={}".format(
                        parameter, value))

            # Create an executable string with given arguments
            full_string = "{}({})".format(
                operator, sep.join(str_parameters))
        else:
            # Create an executable string with default arguments
            full_string = "{}()".format(operator)

        # Store the read operator
        executable_operators.append(full_string)

    # Return two lists of executable operators and selectors
    return executable_operators, selectors
