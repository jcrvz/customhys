# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 14:29:43 2019

@author: Jorge Mario Cruz-Duarte (jcrvz.github.io)
"""
import numpy as np
from itertools import combinations

all_operators = ['random_search', 'random_sample', 'rayleigh_flight',
                 'inertial_pso', 'constricted_pso', 'levy_flight',
                 'mutation_de', 'spiral_dynamic', 'central_force',
                 'gravitational_search', 'ga_mutation', 'ga_crossover',
                 'binomial_crossover_de', 'exponential_crossover_de']

all_selectors = ['greedy', 'probabilistic', 'metropolis', 'all', 'none']

#     Operator call name, dictionary with default parameters, default selector
all_heuristics = [
    ("local_random_walk", dict(probability=0.75, scale=1.0), "greedy"),
    ('random_search', dict(scale=0.01), "greedy"),
    ("random_sample", dict(), "greedy"),
    ("rayleigh_flight", dict(scale=0.01), "greedy"),
    ("levy_flight", dict(scale=1.0, beta=1.5), "greedy"),
    ("mutation_de", dict(scheme=("current-to-best", 1), factor=1.0),
     "greedy"),
    ('binomial_crossover_de', dict(crossover_rate=0.5), "greedy"),
    ("exponential_crossover_de", dict(crossover_rate=0.5), "greedy"),
    ("firefly", dict(epsilon="uniform", alpha=0.8, beta=1.0, gamma=1.0),
     "greedy"),
    ("inertial_pso", dict(inertial=0.7, self_conf=1.54, swarm_conf=1.56),
     "all"),
    ("constriction_pso", dict(kappa=1.0, self_conf=2.54, swarm_conf=2.56),
     "all"),
    ("gravitational_search", dict(alpha=0.02, epsilon=1e-23), "all"),
    ("central_force", dict(gravity=0.001, alpha=0.001, beta=1.5, dt=1.0),
     "all"),
    ("spiral_dynamic", dict(radius=0.9, angle=22.5, sigma=0.1), "all"),
    ("ga_mutation", dict(elite_rate=0.0, mutation_rate=0.2,
                         distribution="uniform", sigma=1.0), "all"),
    ("ga_crossover", dict(pairing="cost", crossover="single",
                          mating_pool_factor=0.1, coefficients=[0.5, 0.5]),
     "all")
]


class Population():
    """
    Generates a population that search along the problem domain using
    different strategies.
    """
    # Internal variables
    iteration = 1
    rotation_matrix = []

    # Parameters per selection method
    metropolis_temperature = 1000.0
    metropolis_rate = 0.01
    metropolis_boltzmann = 1.0
    probability_selection = 0.5

    # Class initialisation
    # -------------------------------------------------------------------------
    def __init__(self, problem_function, boundaries, num_agents=30,
                 is_constrained=True):
        """
        Parameters
        ----------
        problem_function : function
            A function that maps a 1-by-D array of real values ​​to a real
            value.
        boundaries : tuple
            A tuple with two lists of size D corresponding to the lower and
            upper limits of search space, such as:
                boundaries = (lower_boundaries, upper_boundaries)
            Note: Dimensions of search domain are read from these boundaries.
        num_agents : int, optional
            Number of search agents or population size. The default is 30.
        is_constrained : bool, optional
            Avoid agents abandon the search space. The default is True.

        Returns
        -------
        None.

        """
        # Read problem, it must be a callable function
        assert callable(problem_function)
        self.problem_function = problem_function

        # boundaries must be a tuple of np.ndarrays

        # Read number of variables or dimension
        if len(boundaries[0]) == len(boundaries[1]):
            self.num_dimensions = len(boundaries[0])
        else:
            raise self.__PopulationError("Lower and upper boundaries must " +
                                         "have the same length")

        # Read the upper and lower boundaries of search space
        self.lower_boundaries = boundaries[0]
        self.upper_boundaries = boundaries[1]
        self.span_boundaries = self.upper_boundaries - self.lower_boundaries
        self.centre_boundaries = (self.upper_boundaries +
                                  self.lower_boundaries) / 2

        # Read number of agents in population
        assert isinstance(num_agents, int)
        self.num_agents = num_agents

        # Initialise positions and fitness values
        self.positions = np.full((self.num_agents, self.num_dimensions),
                                 np.nan)
        self.velocities = np.full((self.num_agents, self.num_dimensions), 0)
        self.fitness = np.full(self.num_agents, np.nan)

        # General fitness measurements
        self.global_best_position = np.full(self.num_dimensions, np.nan)
        self.global_best_fitness = float('inf')

        self.current_best_position = np.full(self.num_dimensions, np.nan)
        self.current_best_fitness = float('inf')
        self.current_worst_position = np.full(self.num_dimensions, np.nan)
        self.current_worst_fitness = -float('inf')

        self.particular_best_positions = np.full(
            (self.num_agents, self.num_dimensions), np.nan)
        self.particular_best_fitness = np.full(self.num_agents, np.nan)

        self.previous_positions = np.full((self.num_agents,
                                           self.num_dimensions), np.nan)
        self.previous_velocities = np.full((self.num_agents,
                                            self.num_dimensions), np.nan)
        self.previous_fitness = np.full(self.num_agents, np.nan)

        self.is_constrained = is_constrained

        # TODO Add capability for dealing with topologies (neighbourhoods)
        # self.local_best_fitness = self.fitness
        # self.local_best_positions = self.positions

    # ----------------------------------------------------------------------
    #    BASIC TOOLS
    # -------------------------------------------------------------------------

    def get_state(self):
        """
        Generates a string containing the current state of the population

        Returns
        -------
        str
            Information about the best solution found in the current state:
                str = 'x_best = ARRAY, f_best = VALUE'

        """
        return ("x_best = " + str(self.__rescale_back(
            self.global_best_position)) + ", f_best = " +
                str(self.global_best_fitness))

    def get_population(self):
        """
        Gives the current population positions from [-1, 1]^num_dimesions to
        the original search space scale.

        Returns
        -------
        rescaled_positions : ndarray
            Population positions as a num_agents-by-num_dimensions array.

        """
        rescaled_positions = np.tile(
            self.centre_boundaries, (self.num_agents, 1)) + self.positions *\
            np.tile(self.span_boundaries / 2, (self.num_agents, 1))
        return rescaled_positions

    def set_population(self, positions):
        """
        Sets the current population positions from the original search space
        scale to [-1, 1]^num_dimesions.

        Parameters
        ----------
        positions : ndarray
            Population positions as a num_agents-by-num_dimensions array.

        Returns
        -------
        rescaled_positions : ndarray
            Population positions as a num_agents-by-num_dimensions array.

        """
        rescaled_positions = 2 * (positions - np.tile(
            self.centre_boundaries, (self.num_agents, 1))) / np.tile(
            self.span_boundaries, (self.num_agents, 1))

        return rescaled_positions

    def evaluate_fitness(self):
        """
        Evaluates the population positions in the problem function.

        Returns
        -------
        None.

        """
        for agent in range(self.num_agents):
            self.fitness[agent] = self.problem_function(
                self.__rescale_back(self.positions[agent, :]))

    def update_population(self, selection_method="all"):
        """
        Updates the population positions according to a selection scheme.
        The variable all_selectors contains the possible strings.

        Parameters
        ----------
        selection_method : str, optional
            A string corresponding to a selection method. The selectors
            available are: 'greedy', 'probabilistic', 'metropolis', 'all',
            'none'. The default is 'all'.

        Returns
        -------
        None.

        """
        # Update population positons, velocities, and fitness
        for agent in range(self.num_agents):
            if getattr(self, "_selection_" + selection_method)(
                    self.fitness[agent], self.previous_fitness[agent]):
                # if new positions are improved, then update past register ...
                self.previous_fitness[agent] = self.fitness[agent]
                self.previous_positions[agent, :] = self.positions[agent, :]
                self.previous_velocities[agent, :] = self.velocities[agent, :]
            else:
                # ... otherwise,return to previous values
                self.fitness[agent] = self.previous_fitness[agent]
                self.positions[agent, :] = self.previous_positions[agent, :]
                self.velocities[agent, :] = self.previous_velocities[agent, :]

        # Update the current best and worst positions (forced to greedy)
        self.current_best_position = self.positions[self.fitness.argmin(), :]
        self.current_best_fitness = self.fitness.min()
        self.current_worst_position = self.positions[self.fitness.argmax(), :]
        self.current_worst_fitness = self.fitness.max()

    def update_particular(self, selection_method="greedy"):
        """
        Updates the particular positions according to a selection scheme.
        The variable all_selectors contains the possible strings.

        Parameters
        ----------
        selection_method : str, optional
            A string corresponding to a selection method. The selectors
            available are: 'greedy', 'probabilistic', 'metropolis', 'all',
            'none'. The default is 'greedy'.

        Returns
        -------
        None.

        """
        for agent in range(self.num_agents):
            if getattr(self, "_selection_" + selection_method)(
                    self.fitness[agent], self.particular_best_fitness[agent]):
                self.particular_best_fitness[agent] = self.fitness[agent]
                self.particular_best_positions[agent, :] = \
                    self.positions[agent, :]

    def update_global(self, selection_method="greedy"):
        """
        Updates the global position according to a selection scheme.
        The variable all_selectors contains the possible strings.

        Parameters
        ----------
        selection_method : str, optional
            A string corresponding to a selection method. The selectors
            available are: 'greedy', 'probabilistic', 'metropolis', 'all',
            'none'. The default is 'greedy'.

        Returns
        -------
        None.

        """
        # Perform particular updating
        self.update_particular(selection_method)

        # Read current global best agent
        candidate_position = self.particular_best_positions[
                             self.particular_best_fitness.argmin(), :]
        candidate_fitness = self.particular_best_fitness.min()
        if (getattr(self, "_selection_" + selection_method)(
                candidate_fitness, self.global_best_fitness) or
                np.isinf(candidate_fitness)):
            self.global_best_position = candidate_position
            self.global_best_fitness = candidate_fitness

    # -------------------------------------------------------------------------
    #    INITIALISATORS
    # -------------------------------------------------------------------------
    # TODO Add more initialisation operators like grid, boundary, etc.

    def initialise_uniformly(self):
        """
        Initialises population by using a random uniform distribution in
        [-1,1].

        Returns
        -------
        None.
        """

        self.positions = np.random.uniform(-1, 1, (self.num_agents,
                                                   self.num_dimensions))

    # -------------------------------------------------------------------------
    #    PERTURBATORS
    # -------------------------------------------------------------------------

    def random_sample(self):
        """
        Performs a random sampling using a uniform distribution in [-1, 1].

        Returns
        -------
        None.

        """
        # Create random positions using random numbers between -1 and 1
        self.positions = np.random.uniform(-1, 1, (self.num_agents,
                                                   self.num_dimensions))

        # Check constraints
        if self.is_constrained:
            self.__check_simple_constraints()

    def random_search(self, scale=0.01):
        """
        Performs a random walk using a uniform distribution in [-1, 1].

        Parameters
        ----------
        scale : float, optional
            It is the step scale between [0.0, 1.0]. The default is 0.01.

        Returns
        -------
        None.

        """
        # Check the scale value
        self.__check_parameter('scale', scale)

        # Move each agent using uniform random displacements
        self.positions += scale * \
            np.random.uniform(-1, 1, (self.num_agents, self.num_dimensions))

        # Check constraints
        if self.is_constrained:
            self.__check_simple_constraints()

    def rayleigh_flight(self, scale=0.01):
        """
        Perform a Rayleigh flight using a normal standard distribution.

        Parameters
        ----------
        scale : float, optional
            It is the step scale between [0.0, 1.0]. The default is 0.01.

        Returns
        -------
        None.

        """
        # Check the scale value
        self.__check_parameter('scale', scale)

        # Move each agent using gaussian random displacements
        self.positions += scale * \
            np.random.standard_normal((self.num_agents, self.num_dimensions))

        # Check constraints
        if self.is_constrained:
            self.__check_simple_constraints()

    def levy_flight(self, scale=1.0, beta=1.5):
        """
        Perform a Lévy flight by using the Mantegna's algorithm.

        Parameters
        ----------
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
        self.__check_parameter('scale', scale)
        self.__check_parameter('beta', beta, (1.0, 3.0))

        # Calculate x's std dev (Mantegna's algorithm)
        sigma = ((np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2)) /
                 (np.math.gamma((1 + beta) / 2) * beta *
                  (2 ** ((beta - 1) / 2)))) ** (1 / beta)

        # Determine x and y using normal distributions with sigma_y = 1
        x = sigma * np.random.standard_normal((self.num_agents,
                                               self.num_dimensions))
        y = np.abs(np.random.standard_normal((self.num_agents,
                                              self.num_dimensions)))

        # Calculate the random number with levy stable distribution
        levy_random = x / (y ** (1 / beta))

        # Determine z as an additional normal random number
        z = np.random.standard_normal((self.num_agents, self.num_dimensions))

        # Move each agent using levy random displacements
        self.positions += scale * z * levy_random * \
            (self.positions - np.tile(self.global_best_position,
                                      (self.num_agents, 1)))

        # Check constraints
        if self.is_constrained:
            self.__check_simple_constraints()

    def inertial_pso(self, inertial=0.7, self_conf=1.54, swarm_conf=1.56):
        """
        Performs a swarm movement by using the inertial version of Particle
        Swarm Optimisation (PSO).

        Parameters
        ----------
        inertial : float, optional
            Inertial factor. The default is 0.7.
        self_conf : float, optional
            Self confidence factor. The default is 1.54.
        swarm_conf : float, optional
            Swarm confidence factor. The default is 1.56.

        Returns
        -------
        None.

        """
        # Check the scale and beta value
        self.__check_parameter('inertial', inertial)
        self.__check_parameter('self_conf', self_conf, (0.0, 10.0))
        self.__check_parameter('swarm_conf', swarm_conf, (0.0, 10.0))

        # Determine random numbers
        r_1 = self_conf * np.random.rand(self.num_agents, self.num_dimensions)
        r_2 = swarm_conf * np.random.rand(self.num_agents, self.num_dimensions)

        # Find new velocities
        self.velocities = inertial * self.velocities + r_1 * (
                self.particular_best_positions - self.positions) + \
            r_2 * (np.tile(self.global_best_position, (self.num_agents, 1)) -
                   self.positions)

        # Move each agent using velocity's information
        self.positions += self.velocities

        # Check constraints
        if self.is_constrained:
            self.__check_simple_constraints()

    def constriction_pso(self, kappa=1.0, self_conf=2.54, swarm_conf=2.56):
        """
        Performs a swarm movement by using the constricted version of Particle
        Swarm Optimisation (PSO).

        Parameters
        ----------
        kappa : float, optional
            Kappa factor. The default is 0.7.
        self_conf : float, optional
            Self confidence factor. The default is 1.54.
        swarm_conf : float, optional
            Swarm confidence factor. The default is 1.56.

        Returns
        -------
        None.

        """
        # Check the scale and beta value
        self.__check_parameter('kappa', kappa)
        self.__check_parameter('self_conf', self_conf, (0.0, 10.0))
        self.__check_parameter('swarm_conf', swarm_conf, (0.0, 10.0))

        # Find the constriction factor chi using phi
        phi = self_conf + swarm_conf
        if phi > 4:
            chi = 2 * kappa / np.abs(2 - phi - np.sqrt(phi ** 2 - 4 * phi))
        else:
            chi = np.sqrt(kappa)

        # Determine random numbers
        r_1 = self_conf * np.random.rand(self.num_agents, self.num_dimensions)
        r_2 = swarm_conf * np.random.rand(self.num_agents, self.num_dimensions)

        # Find new velocities
        self.velocities = chi * (self.velocities + r_1 * (
            self.particular_best_positions - self.positions) +
            r_2 * (np.tile(self.global_best_position, (self.num_agents, 1)) -
                   self.positions))

        # Move each agent using velocity's information
        self.positions += self.velocities

        # Check constraints
        if self.is_constrained:
            self.__check_simple_constraints()

    def mutation_de(self, expression="current-to-best", num_rands=1,
                    factor=1.0):
        """
        Mutates the population positions using Differential Evolution (DE)

        Parameters
        ----------
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
        assert isinstance(num_rands, int)
        self.__check_parameter('factor', factor)

        # Create mutants using the expression provided in scheme
        if expression == "rand":
            mutant = self.positions[np.random.permutation(self.num_agents), :]

        elif expression == "best":
            mutant = np.tile(self.global_best_position, (self.num_agents, 1))

        elif expression == "current":
            mutant = self.positions

        elif expression == "current-to-best":
            mutant = self.positions + factor * \
                (np.tile(self.global_best_position,
                         (self.num_agents, 1)) -
                 self.positions[np.random.permutation(self.num_agents), :])

        elif expression == "rand-to-best":
            mutant = self.positions[np.random.permutation(self.num_agents),
                                    :] + factor * \
                (np.tile(self.global_best_position, (self.num_agents, 1)) -
                 self.positions[np.random.permutation(self.num_agents), :])

        elif expression == "rand-to-bestandcurrent":
            mutant = self.positions[np.random.permutation(
                self.num_agents), :] + factor * (np.tile(
                    self.global_best_position, (self.num_agents, 1)) -
                    self.positions[np.random.permutation(
                        self.num_agents), :] + self.positions[
                            np.random.permutation(
                                self.num_agents), :] - self.positions)
        else:
            mutant = []
            raise self.__PopulationError('Invalid DE mutation scheme!')

        # Add random parts according to num_rands
        if num_rands >= 0:
            for _ in range(num_rands):
                mutant += factor * (self.positions[np.random.permutation(
                    self.num_agents), :] - self.positions[
                        np.random.permutation(self.num_agents), :])
        else:
            raise self.__PopulationError('Invalid DE mutation scheme!')

        # Replace mutant population in the current one
        self.positions = mutant

        # Check constraints
        if self.is_constrained:
            self.__check_simple_constraints()

    def binomial_crossover_de(self, crossover_rate=0.5):
        """
        Performs the binomial crossover from Differential Evolution (DE).

        Parameters
        ----------
        crossover_rate : float, optional
            Probability factor to perform the crossover. The default is 0.5.

        Returns
        -------
        None.

        """
        # Check the scale and beta value
        self.__check_parameter('crossover_rate', crossover_rate)

        # Define indices
        indices = np.tile(np.arange(self.num_dimensions), (self.num_agents, 1))

        # Permute indices per dimension
        rand_indices = np.vectorize(np.random.permutation,
                                    signature='(n)->(n)')(indices)

        # Calculate the NOT condition (because positions were already updated!)
        condition = np.logical_not((indices == rand_indices) | (
            np.random.rand(self.num_agents, self.num_dimensions) <=
            crossover_rate))

        # Reverse the ones to their previous positions
        self.positions[condition] = self.previous_positions[condition]

        # Check constraints
        if self.is_constrained:
            self.__check_simple_constraints()

    def exponential_crossover_de(self, crossover_rate=0.5):
        """
        Performs the exponential crossover from Differential Evolution (DE)

        Parameters
        ----------
        crossover_rate : float, optional
            Probability factor to perform the crossover. The default is 0.5.

        Returns
        -------
        None.

        """
        # Check the scale and beta value
        self.__check_parameter('crossover_rate', crossover_rate)

        # Perform the exponential crossover procedure
        for agent in range(self.num_agents):
            for dim in range(self.num_dimensions):
                # Initialise L and choose a random index n
                exp_var = 0
                n = np.random.randint(self.num_dimensions)
                while True:
                    # Increase L and check the exponential crossover condition
                    exp_var += 1
                    if np.logical_not((np.random.rand() < crossover_rate) and
                                      (exp_var < self.num_dimensions)):
                        break

                # Perform the crossover if the following condition is met
                if dim not in [(n + x) % self.num_dimensions for x in
                               range(exp_var)]:
                    self.positions[agent, dim] = self.previous_positions[
                        agent, dim]

        # Check constraints
        if self.is_constrained:
            self.__check_simple_constraints()

    def local_random_walk(self, probability=0.75, scale=1.0):
        """
        Performs the local random walk from Cuckoo Search (CS)

        Parameters
        ----------
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
        self.__check_parameter('probability', probability)
        self.__check_parameter('scale', scale)

        # Determine random numbers
        r_1 = np.random.rand(self.num_agents, self.num_dimensions)
        r_2 = np.random.rand(self.num_agents, self.num_dimensions)

        # Move positions with a displacement due permutations and probabilities
        self.positions += scale * r_1 * (self.positions[
            np.random.permutation(self.num_agents), :] - self.positions[
                np.random.permutation(self.num_agents), :]) * np.heaviside(
                    r_2 - probability, 0)

        # Check constraints
        if self.is_constrained:
            self.__check_simple_constraints()

    def spiral_dynamic(self, radius=0.9, angle=22.5, sigma=0.1):
        """
        Performs the deterministic or stochastic spiral dynamic movement

        Parameters
        ----------
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
        self.__check_parameter('radius', radius)
        self.__check_parameter('angle', angle, (0.0, 360.0))
        self.__check_parameter('sigma', sigma)

        # Update rotation matrix
        self.__get_rotation_matrix(np.deg2rad(angle))

        for agent in range(self.num_agents):
            random_radii = np.random.uniform(radius - sigma, radius + sigma,
                                             self.num_dimensions)
            # If random radii need to be constrained to [0, 1]:
            self.positions[agent, :] = self.global_best_position + \
                random_radii * np.matmul(
                    self.rotation_matrix, (self.positions[agent, :] -
                                           self.global_best_position))

        # Check constraints
        if self.is_constrained:
            self.__check_simple_constraints()

    # Firefly (generalised)
    # -------------------------------------------------------------------------
    def firefly(self, epsilon="uniform", alpha=0.8, beta=1.0, gamma=1.0):
        """
        Performs movements accordint to the Firefly algorithm (FA)

        Parameters
        ----------
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
        # Check the scale and beta value
        self.__check_parameter('alpha', alpha)
        self.__check_parameter('beta', beta)
        self.__check_parameter('gamma', gamma)

        # Determine epsilon values
        if epsilon == "gaussian":
            epsilon_value = np.random.standard_normal(
                (self.num_agents, self.num_dimensions))

        elif epsilon == "uniform":
            epsilon_value = np.random.uniform(
                -0.5, 0.5, (self.num_agents, self.num_dimensions))
        else:
            epsilon_value = []
            raise self.__PopulationError("Epsilon is not valid: 'uniform' " +
                                         " or 'gaussian'")

        # Initialise delta or difference between two positions
        difference_positions = np.zeros((self.num_agents, self.num_dimensions))

        for agent in range(self.num_agents):
            # Select indices in order to avoid division by zero
            indices = (np.arange(self.num_agents) != agent)

            # Determine all vectorial distances with respect to agent
            delta = self.positions[indices, :] - np.tile(
                self.positions[agent, :], (self.num_agents - 1, 1))

            # Determine differences between lights
            delta_lights = np.tile(
                (self.fitness[indices] - np.tile(
                    self.fitness[agent], (1, self.num_agents-1))).transpose(),
                (1, self.num_dimensions))

            # Find the total attraction for each agent
            difference_positions[agent, :] = np.sum(
                np.heaviside(-delta_lights, 0) * delta *
                np.exp(-gamma * np.linalg.norm(delta, 2, 1) ** 2), 0)

        # Move fireflies according to their attractions
        self.positions += alpha * epsilon_value + beta * difference_positions

        # Check constraints
        if self.is_constrained:
            self.__check_simple_constraints()

    # Central Force Optimisation (CFO)
    # -------------------------------------------------------------------------
    def central_force(self, gravity=1.0, alpha=0.5, beta=1.5, dt=1.0):
        # Initialise acceleration
        acceleration = np.zeros((self.num_agents, self.num_dimensions))

        for agent in range(self.num_agents):
            # Select indices in order to avoid division by zero
            indices = (np.arange(self.num_agents) != agent)

            # Determine all masses differences with respect to agent
            delta_masses = self.fitness[indices] - np.tile(
                self.fitness[agent], (1, self.num_agents - 1))

            # Determine all vectorial distances with respect to agent
            delta_positions = self.positions[indices, :] - np.tile(
                self.positions[agent, :], (self.num_agents - 1, 1))

            distances = np.linalg.norm(delta_positions, 2, 1)

            # Find the quotient part    ! -> - delta_masses (cz minimisation)
            quotient = np.heaviside(-delta_masses, 0) * (
                np.abs(delta_masses) ** alpha) / (distances ** beta)

            # Determine the acceleration for each agent
            acceleration[agent, :] = gravity * np.sum(
                delta_positions * np.tile(quotient.transpose(),
                                          (1, self.num_dimensions)), 0)

        self.positions += 0.5 * acceleration * (dt ** 2)

        # Check constraints
        if self.is_constrained:
            self.__check_simple_constraints()

    # Gravitational Search Algorithm (GSA) simplified
    # -------------------------------------------------------------------------
    def gravitational_search(self, gravity=1.0, alpha=0.5, epsilon=1e-23):
        # Initialise acceleration
        acceleration = np.zeros((self.num_agents, self.num_dimensions))

        # Determine the gravitational constant
        gravitation = gravity * np.exp(- alpha * self.iteration)

        # Determine mass for each agent
        raw_masses = (self.fitness - np.tile(self.current_worst_fitness,
                                             (1, self.num_agents)))
        masses = (raw_masses / np.sum(raw_masses)).reshape(self.num_agents)

        for agent in range(self.num_agents):
            # Select indices in order to avoid division by zero
            indices = (np.arange(self.num_agents) != agent)

            # Determine all vectorial distances with respect to agent
            delta_positions = self.positions[indices, :] - np.tile(
                self.positions[agent, :], (self.num_agents - 1, 1))

            quotient = masses[indices] / (
                np.linalg.norm(delta_positions, 2, 1) + epsilon)

            # Force interaction
            force_interaction = gravitation * np.tile(
                quotient.reshape(self.num_agents - 1, 1),
                (1, self.num_dimensions)) * delta_positions

            # Acceleration
            acceleration[agent, :] = np.sum(np.random.rand(
                self.num_agents - 1, self.num_dimensions) *
                force_interaction, 0)

        # Update velocities
        self.velocities = acceleration + np.random.rand(
            self.num_agents, self.num_dimensions) * self.velocities

        # Update positions
        self.positions += self.velocities

        # Check constraints
        if self.is_constrained:
            self.__check_simple_constraints()

    # Genetic Algorithm (GA): Crossover
    # -------------------------------------------------------------------------
    def ga_crossover(self, pairing="cost", crossover="single",
                     mating_pool_factor=0.1):
        # Mating pool size
        num_mates = int(np.round(mating_pool_factor * self.num_agents))

        # Number of offsprings
        num_offsprings = self.num_agents - num_mates

        # Get the mating pool using the selection strategy specified
        mating_pool_indices = self._ga_natural_selection(num_mates)

        # Get parents (at least a couple per offspring)
        if len(pairing) > 10:  # if pairing = 'tournament_2_100', for example
            pairing, tournament_size, tournament_probability = \
                pairing.split("_")
        else:  # dummy (it must not be used)
            tournament_size, tournament_probability = '-1.5', '-1'
        couple_indices = getattr(self, "_ga_" + pairing + "_pairing")(
                    mating_pool_indices, num_offsprings, int(tournament_size),
                    float(tournament_probability)/100)

        # Identify offspring indices
        offspring_indices = np.setdiff1d(
            np.arange(self.num_agents), mating_pool_indices, True)

        # Perform crossover and assign to population
        if len(crossover) > 7:  # if crossover = 'linear_0.5_0.5', for example
            crossover, coeff1, coeff2 = crossover.split("_")
            coefficients = [float(coeff1), float(coeff2)]
        else:  # dummy (it must not be used)
            coefficients = [np.nan, np.nan]
        self.positions[offspring_indices, :] = getattr(
            self, "_ga_" + crossover + "_crossover")(
                couple_indices.astype(np.int64), coefficients)

    # Genetic Algorithm (GA): Selection strategies
    # -------------------------------------------------------------------------
    # -> Natural selection to obtain the mating pool
    def _ga_natural_selection(self, num_mates):
        # Sort population according to its fitness values
        sorted_indices = np.argsort(self.fitness)

        # Return indices corresponding mating pool
        return sorted_indices[:num_mates]

    # Genetic Algorithm (GA): Pairing strategies
    # -------------------------------------------------------------------------
    # -> Even-and-Odd pairing
    def _ga_evenodd_pairing(self, mating_pool, num_couples, *args):
        # Check if the num of mates is even
        mating_pool_size = mating_pool.size - (mating_pool.size % 2)
        half_size = mating_pool_size // 2

        # Dummy indices according to the mating pool size
        remaining = num_couples - half_size
        if remaining > 0:
            dummy_indices = np.tile(
                np.reshape(np.arange(mating_pool_size), (-1, 2)).transpose(),
                (1, int(np.ceil(num_couples / half_size))))
        else:
            dummy_indices = np.reshape(np.arange(mating_pool_size),
                                       (-1, 2)).transpose()

        # Return couple_indices
        return mating_pool[dummy_indices[:, :num_couples]]

    # -> Random pairing
    def _ga_random_pairing(self, mating_pool, num_couples, *args):
        # Return two random indices from mating pool
        return mating_pool[np.random.randint(mating_pool.size,
                                             size=(2, num_couples))]

    # -> Tournament pairing
    def _ga_tournament_pairing(self, mating_pool, num_couples,
                               tournament_size=2, probability=1.0):
        # Calculate probabilities
        probabilities = probability * (
            (1 - probability) ** np.arange(tournament_size))

        # Initialise the mother and father indices
        couple_indices = np.full((2, num_couples), np.nan)

        # Perform tournaments until all mates are selected
        for couple in range(num_couples):
            mate = 0
            while mate < 2:
                # Choose tournament candidates
                random_indices = mating_pool[
                    np.random.permutation(mating_pool.size)[:tournament_size]]

                # Determine the candidate fitness values
                candidates_indices = random_indices[
                    np.argsort(self.fitness[random_indices])]

                # Find the best according to its fitness and probability
                winner = candidates_indices[
                    np.random.rand(tournament_size) < probabilities]
                if winner.size > 0:
                    couple_indices[mate, couple] = int(winner[0])
                    mate += 1

        # Return the mating pool and its fitness
        return couple_indices

    # -> Roulette Wheel (Rank Weighting) Selection
    def _ga_rank_pairing(self, mating_pool, num_couples, *args):
        # Determine the probabilities
        probabilities = (mating_pool.size - np.arange(mating_pool.size)) /\
            np.sum(np.arange(mating_pool.size) + 1)

        # Perform the roulette wheel selection and return couples
        couple_indices = np.searchsorted(
            np.cumsum(probabilities), np.random.rand(2*num_couples))

        # Return couples
        return couple_indices.reshape((2, -1))

    # -> Roulette Wheel (Cost Weighting) Selection
    def _ga_cost_pairing(self, mating_pool, num_couples, *args):
        # Cost normalisation from mating pool: cost - min(cost @ non mates)
        normalised_cost = self.fitness[mating_pool] - np.min(self.fitness[
            np.setdiff1d(np.arange(self.num_agents), mating_pool)])

        # Determine the related probabilities
        probabilities = np.abs(normalised_cost / np.sum(normalised_cost))

        # Perform the roulette wheel selection and return couples
        couple_indices = np.searchsorted(np.cumsum(probabilities),
                                         np.random.rand(2*num_couples))

        # Return couples
        return couple_indices.reshape((2, -1))

    # Genetic Algorithm (GA): Crossover strategies
    # -------------------------------------------------------------------------
    # -> Single-Point Crossover
    def _ga_single_crossover(self, parent_indices, *args):
        # Determine the single point per each couple
        single_points = np.tile(np.random.randint(
            self.num_dimensions, size=parent_indices.shape[1]),
            (self.num_dimensions, 1)).transpose()

        # Crossover condition mask
        crossover_mask = np.tile(np.arange(self.num_dimensions),
                                 (parent_indices.shape[1], 1)) <= single_points

        # Get father and mother
        father_position = self.positions[parent_indices[0, :], :]
        mother_position = self.positions[parent_indices[1, :], :]

        # Initialise offsprings with mother positions
        offsprings = mother_position
        offsprings[crossover_mask] = father_position[crossover_mask]

        # Return offspring positions
        return offsprings

    # -> Two-Point Crossover
    # !!! It can be extended to multiple points
    def _ga_two_crossover(self, parent_indices, *args):
        # Find raw points
        raw_points = np.sort(np.random.randint(
            self.num_dimensions, size=(parent_indices.shape[1], 2)))

        # Determine the single point per each couple
        points = [np.tile(raw_points[:, x], (
            self.num_dimensions, 1)).transpose()
            for x in range(raw_points.shape[1])]

        # Range matrix
        dummy_matrix = np.tile(np.arange(self.num_dimensions),
                               (parent_indices.shape[1], 1))

        # Crossover condition mask (only for two points)
        crossover_mask = ((dummy_matrix <= points[0]) |
                          (dummy_matrix > points[1]))
        # OPTIMIZE : crossover_mask = points[1] < dummy_matrix <= points[0]

        # Get father and mother
        father_position = self.positions[parent_indices[0, :], :]
        mother_position = self.positions[parent_indices[1, :], :]

        # Initialise offsprings with mother positions
        offsprings = mother_position
        offsprings[crossover_mask] = father_position[crossover_mask]

        # Return offspring positions
        return offsprings

    # -> Uniform Crossover
    def _ga_uniform_crossover(self, parent_indices, *args):
        # Crossover condition mask (only for two points)
        crossover_mask = np.random.rand(
            parent_indices.shape[1], self.num_dimensions) < 0.5

        # Get father and mother
        father_position = self.positions[parent_indices[0, :], :]
        mother_position = self.positions[parent_indices[1, :], :]

        # Initialise offsprings with mother positions
        offsprings = mother_position
        offsprings[crossover_mask] = father_position[crossover_mask]

        # Return offspring positions
        return offsprings

    # -> Random blending crossover
    def _ga_blend_crossover(self, parent_indices, *args):
        # Initialise random numbers between 0 and 1
        beta_values = np.random.rand(
            parent_indices.shape[1], self.num_dimensions)

        # Get father and mother
        father_position = self.positions[parent_indices[0, :], :]
        mother_position = self.positions[parent_indices[1, :], :]

        # Determine offsprings with father and mother positions
        offsprings = beta_values * father_position +\
            (1 - beta_values) * mother_position

        # Return offspring positions
        return offsprings

    # -> Linear crossover: ofspring = coeff[0] * father + coeff[1] * mother
    def _ga_linear_crossover(self, parent_indices, coefficients=[0.5, 0.5]):
        # Get father and mother
        father_position = self.positions[parent_indices[0, :], :]
        mother_position = self.positions[parent_indices[1, :], :]

        # Determine offsprings with father and mother positions
        offsprings = coefficients[0] * father_position +\
            coefficients[1] * mother_position

        # Return offspring positions
        return offsprings

    # Genetic Algorithm (GA): Selection strategies
    # -------------------------------------------------------------------------
    # -> Natural selection to obtain the mating pool
    def ga_mutation(self, elite_rate=0.0, mutation_rate=0.2,
                    distribution="uniform", sigma=1.0):
        num_elite = int(np.ceil(self.num_agents * elite_rate))

        # If num_elite equals num_agents then do nothing, or ...
        if (num_elite < self.num_agents):
            # Number of mutations to perform
            num_mutations = int(np.round(self.num_agents * self.num_dimensions
                                         * mutation_rate))

            # Identify mutable agents
            dimension_indices = np.random.randint(
                0, self.num_dimensions, num_mutations)

            if num_elite > 0:
                agent_indices = np.argsort(self.fitness)[
                    np.random.randint(num_elite, self.num_agents,
                                      num_mutations)]
            else:
                agent_indices = np.random.randint(num_elite, self.num_agents,
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
            self.positions[rows.flatten(), columns.flatten()] = mutants

    # ----------------------------------------------------------------------
    #    INTERNAL METHODS (avoid using them outside)
    # -------------------------------------------------------------------------

    # Check simple constraints if self.is_constrained = True
    # -------------------------------------------------------------------------
    def __check_simple_constraints(self):
        # Check if agents are beyond lower boundaries
        low_check = self.positions < -1.
        if low_check.any():
            # Fix them
            self.positions[low_check] = -1.
            self.velocities[low_check] = 0.

        # Check if agents are beyond upper boundaries
        upp_check = self.positions > 1.
        if upp_check.any():
            # Fix them
            self.positions[upp_check] = 1.
            self.velocities[upp_check] = 0.

    # Rescale an agent from [-1,1] to [lower,upper] per dimension
    # -------------------------------------------------------------------------
    def __rescale_back(self, position):
        return self.centre_boundaries + position * (self.span_boundaries / 2)

    # Generate a N-D rotation matrix for a given angle
    # -------------------------------------------------------------------------
    def __get_rotation_matrix(self, angle=0.39269908169872414):
        # Initialise the rotation matrix
        rotation_matrix = np.eye(self.num_dimensions)

        # Find the combinations without repetions
        planes = list(combinations(range(self.num_dimensions), 2))

        # Create the rotation matrix
        for xy in range(len(planes)):
            # Read dimensions
            x, y = planes[xy]

            # (Re)-initialise a rotation matrix for each plane
            rotation_plane = np.eye(self.num_dimensions)

            # Assign corresponding values
            rotation_plane[x, y] = np.cos(angle)
            rotation_plane[y, y] = np.cos(angle)
            rotation_plane[x, y] = -np.sin(angle)
            rotation_plane[y, x] = np.sin(angle)

            rotation_matrix = np.matmul(rotation_matrix, rotation_plane)

        self.rotation_matrix = rotation_matrix

    # TODO: Selection in one method - Fix other codes
    def _selection(self, new, old, selector="greedy"):

        # Greedy selection
        if selector == "greedy":
            selection_condition = new <= old
        #
        # Metropolis selection
        elif selector == "metropolis":
            if new <= old:
                selection_condition = True
            else:
                if np.math.exp(-(new - old) / (
                        self.metropolis_boltzmann *
                        self.metropolis_temperature * (
                            (1 - self.metropolis_rate) ** self.iteration) +
                        1e-23)) > np.random.rand():
                    selection_condition = True
                else:
                    selection_condition = False
        #
        # Probabilistic selection
        elif selector == "probabilistic":
            if new <= old:
                selection_condition = True
            else:
                if np.random.rand() < self.probability_selection:
                    selection_condition = True
                else:
                    selection_condition = False
        #
        # All selection
        elif selector == "all":
            selection_condition = True
        #
        # None selection
        elif selector == "none":
            selection_condition = False

        return selection_condition

    # Greedy selection : new is better than old one
    # -------------------------------------------------------------------------
    def _selection_greedy(self, new, old):
        return new <= old

    # Metropolis selection : apply greedy selection and worst with a prob
    # -------------------------------------------------------------------------
    def _selection_metropolis(self, new, old):
        # It depends of metropolis_temperature, metropolis_rate, and iteration
        if new <= old:
            selection_condition = True
        else:
            if np.math.exp(-(new - old) / (
                    self.metropolis_boltzmann * self.metropolis_temperature *
                    ((1 - self.metropolis_rate) ** self.iteration) + 1e-23)
                    ) > np.random.rand():
                selection_condition = True
            else:
                selection_condition = False
        return selection_condition

        # Probabilistic selection: worst is chosen iif rand < prob

    # -------------------------------------------------------------------------
    def _selection_probabilistic(self, new, old):
        # It depends of metropolis_temperature, metropolis_rate, and iteration
        if new <= old:
            selection_condition = True
        else:
            if np.random.rand() < self.probability_selection:
                selection_condition = True
            else:
                selection_condition = False
        return selection_condition

        # All selection : only new does matter

    # -------------------------------------------------------------------------
    def _selection_all(self, *args):
        return True

    # None selection : new does not matter
    # -------------------------------------------------------------------------
    def _selection_none(self, *args):
        return False

    # Additional internal methods
    class __PopulationError(Exception):
        """
        Simple PopulationError to manage exceptions.
        """
        pass

    def __check_parameter(self, par_name, par_value, interval=(0.0, 1.0),
                          par_type=float):
        """
        Check if a parameter or variable is into an interval.

        Parameters
        ----------
        par_name: str
            Variable's name to print the error message (if so)
        par_value : int, float
            Variable to check.
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
        # Check if the parameter value is into the interval
        assert isinstance(par_value, par_type)
        if not interval[0] <= par_value <= interval[1]:
            raise self.__PopulationError(f"Invalid value! {par_name} = " +
                                         f"{par_value} must be in " +
                                         f"[{interval[0]}, {interval[1]}]")
