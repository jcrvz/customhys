# -*- coding: utf-8 -*-
"""
This module contains the class Population.

Created on Tue Sep 17 14:29:43 2019

@author: Jorge Mario Cruz-Duarte (jcrvz.github.io), e-mail: jorge.cruz@tec.mx
"""

import numpy as np

__all__ = ['Population']
__selectors__ = ['all', 'greedy', 'metropolis', 'probabilistic']

class Population:
    """
    This is the Population class, each object corresponds to a population of agents within a search space.
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
    # ------------------------------------------------------------------------
    def __init__(self, boundaries, num_agents=30, is_constrained=True):
        """
        Return a population of size ``num_agents`` within a problem domain defined by ``boundaries``.

        :param tuple boundaries:
            A tuple with two lists of size D corresponding to the lower and upper limits of search space, such as:
                boundaries = (lower_boundaries, upper_boundaries)
            Note: Dimensions of search domain are read from these boundaries.
        :param int num_agents: optional.
            Number of search agents or population size. The default is 30.
        :param bool is_constrained: optional.
            Avoid agents abandon the search space. The default is True.

        :returns: population object.
        """
        # Read number of variables or dimension
        if len(boundaries[0]) == len(boundaries[1]):
            self.num_dimensions = len(boundaries[0])
        else:
            raise PopulationError('Lower and upper boundaries must have the same length')

        # Read the upper and lower boundaries of search space
        self.lower_boundaries = boundaries[0]
        self.upper_boundaries = boundaries[1]
        self.span_boundaries = self.upper_boundaries - self.lower_boundaries
        self.centre_boundaries = (self.upper_boundaries + self.lower_boundaries) / 2.

        # Read number of agents in population
        assert isinstance(num_agents, int)
        self.num_agents = num_agents

        # Initialise positions and fitness values
        self.positions = np.full((self.num_agents, self.num_dimensions), np.nan)
        self.velocities = np.full((self.num_agents, self.num_dimensions), 0)
        self.fitness = np.full(self.num_agents, np.nan)

        # General fitness measurements
        self.global_best_position = np.full(self.num_dimensions, np.nan)
        self.global_best_fitness = float('inf')

        self.current_best_position = np.full(self.num_dimensions, np.nan)
        self.current_best_fitness = float('inf')
        self.current_worst_position = np.full(self.num_dimensions, np.nan)
        self.current_worst_fitness = -float('inf')

        self.particular_best_positions = np.full((self.num_agents, self.num_dimensions), np.nan)
        self.particular_best_fitness = np.full(self.num_agents, np.nan)

        self.previous_positions = np.full((self.num_agents, self.num_dimensions), np.nan)
        self.previous_velocities = np.full((self.num_agents, self.num_dimensions), np.nan)
        self.previous_fitness = np.full(self.num_agents, np.nan)

        self.is_constrained = is_constrained

        # TODO Add capability for dealing with topologies (neighbourhoods)
        # self.local_best_fitness = self.fitness
        # self.local_best_positions = self.positions

    # ===========
    # BASIC TOOLS
    # ===========

    def get_state(self):
        """
        Return a string containing the current state of the population, i.e.,
            str = 'x_best = ARRAY, f_best = VALUE'

        :returns: str
        """
        return ('x_best = ' + str(self._rescale_back(self.global_best_position)) +
                ', f_best = ' + str(self.global_best_fitness))

    def get_positions(self):
        """
        Return the current population positions. Positions are represented in a matrix of size:
            ``positions.shape() = (num_agents, num_dimensions)``

        **NOTE:** The position is rescaled from the normalised search space, i.e., [-1, 1]^num_dimensions.

        :returns: numpy.ndarray
        """
        return np.tile(self.centre_boundaries, (self.num_agents, 1)) + \
               self.positions * np.tile(self.span_boundaries / 2., (self.num_agents, 1))

    def set_positions(self, positions):
        """
        Return the current population positions. Positions are represented in a matrix of size:
            ``positions.shape() = (num_agents, num_dimensions)``

        Note: The position is rescaled to the original search space.

        :param numpy.ndarray positions:
            Population positions must have the size num_agents-by-num_dimensions array.

        :returns: numpy.ndarray
        """
        return 2. * (positions - np.tile(self.centre_boundaries, (self.num_agents, 1))) / np.tile(
            self.span_boundaries, (self.num_agents, 1))

    def update_positions(self, level='population', selector='all'):
        """
        Update the population positions according to the level and selection scheme.

        **NOTE:** When an operator is applied (from the operators' module), it automatically replaces new positions, so
        the logic of selectors is contrary as they commonly do.

        :param str level: optional
            Update level, it can be 'population' for the entire population, 'particular' for each agent (an its
            historical performance), and 'global' for the current solution. The default is 'population'.
        :param str selector: optional
            Selection method. The selectors available are: 'greedy', 'probabilistic', 'metropolis', 'all', and 'none'.
            The default is 'all'.

        :returns: None.
        """
        # Update population positons, velocities and fitness
        if level == 'population':
            for agent in range(self.num_agents):
                if self._selection(self.fitness[agent], self.previous_fitness[agent], selector):
                    # if new positions are improved, then update past register
                    self.previous_fitness[agent] = np.copy(self.fitness[agent])
                    self.previous_positions[agent, :] = np.copy(self.positions[agent, :])
                    self.previous_velocities[agent, :] = np.copy(self.velocities[agent, :])
                else:
                    # ... otherwise,return to previous values
                    self.fitness[agent] = np.copy(self.previous_fitness[agent])
                    self.positions[agent, :] = np.copy(self.previous_positions[agent, :])
                    self.velocities[agent, :] = np.copy(self.previous_velocities[agent, :])

            # Update the current best and worst positions (forced to greedy)
            self.current_best_position = np.copy(self.positions[self.fitness.argmin(), :])
            self.current_best_fitness = np.min(self.fitness)
            self.current_worst_position = np.copy(self.positions[self.fitness.argmax(), :])
            self.current_worst_fitness = np.min(self.fitness)

        # Update particular positions, velocities and fitness
        elif level == 'particular':
            for agent in range(self.num_agents):
                if self._selection(self.fitness[agent], self.particular_best_fitness[agent], selector):
                    self.particular_best_fitness[agent] = np.copy(self.fitness[agent])
                    self.particular_best_positions[agent, :] = np.copy(self.positions[agent, :])

        # Update global positions, velocities and fitness
        elif level == 'global':
            # Perform particular updating (recursive)
            self.update_positions('particular', selector)

            # Read current global best agent
            candidate_position = np.copy(self.particular_best_positions[self.particular_best_fitness.argmin(), :])
            candidate_fitness = np.min(self.particular_best_fitness)
            if self._selection(candidate_fitness, self.global_best_fitness, selector) or np.isinf(candidate_fitness):
                self.global_best_position = np.copy(candidate_position)
                self.global_best_fitness = np.copy(candidate_fitness)

        # Raise an error
        else:
            PopulationError('Invalid update level')

    def evaluate_fitness(self, problem_function):
        """
        Evaluate the population positions in the problem function.

        :param function problem_function:
            A function that maps a 1-by-D array of real values ​​to a real value.

        :returns: None.
        """
        # Read problem, it must be a callable function
        assert callable(problem_function)

        # Check simple constraints before evaluate
        if self.is_constrained:
            self._check_simple_constraints()

        # Evaluate each agent in this function
        for agent in range(self.num_agents):
            self.fitness[agent] = problem_function(self._rescale_back(self.positions[agent, :]))

    # ==============
    # INITIALISATORS
    # ==============
    # TODO Add more initialisation operators like grid, boundary, etc.

    def initialise_positions(self, scheme='random'):
        """
        Initialise population by an initialisation scheme.

        :param str scheme: optional
            Initialisation scheme. Right now, there is only 'random' initialisation, which consists of using a random
            uniform distribution in [-1,1]. The default is 'random'.

        :returns: None.
        """
        self.positions = np.random.uniform(-1, 1, (self.num_agents, self.num_dimensions))

    # ================
    # INTERNAL METHODS
    # ================
    # Avoid using them outside

    def _check_simple_constraints(self):
        """
        Check simple constraints for all the dimensions like:
            -1 <= position <= 1, for all i in 1, 2, ..., num_dimensions
        When an agent position is outside the search space, it is reallocated at the closest boundary and its velocity
        is set zero (if so).

        **NOTE:** This check is performed only if Population.is_constrained = True.

        :returns: None.
        """
        # Check if there are nans values
        if np.any(np.isnan(self.positions)):
            np.nan_to_num(self.positions, copy=False, nan=1.0, posinf=1.0, neginf=-1.0)

        # Check if agents are beyond lower boundaries
        low_check = np.less(self.positions, -1.0)
        if np.any(low_check):
            # Fix them
            self.positions[low_check] = -1.0
            self.velocities[low_check] = 0.0

        # Check if agents are beyond upper boundaries
        upp_check = np.greater(self.positions, 1.0)
        if np.any(upp_check):
            # Fix them
            self.positions[upp_check] = 1.0
            self.velocities[upp_check] = 0.0

    def _rescale_back(self, position):
        """
        Rescale an agent position from [-1.0, 1.0] to the original search space boundaries per dimension.

        :param numpy.ndarray position:
            A position given by an array of 1-by-D with elements between [-1, 1].

        :returns: ndarray
        """
        return self.centre_boundaries + position * (self.span_boundaries / 2)

    def _selection(self, new, old, selector='greedy'):
        """
        Answer the question: 'should this new position be accepted?' To do so, a selection procedure is applied.

        :param numpy.ndarray new:
            A new position given by an array of 1-by-num_dimensions with elements between [-1, 1].
        :param numpy.ndarray old:
            An old position given by an array of 1-by-num_dimensions with elements between [-1, 1].
        :param str selector: optional
            A selection scheme used for deciding if the new position is kept. The default is 'greedy'.

        :returns: bool
        """
        # Greedy selection
        if selector == 'greedy':
            selection_condition = new <= old

        # Metropolis selection
        elif selector == 'metropolis':
            if new <= old:
                selection_condition = True
            else:
                selection_condition = bool(np.math.exp(-(new - old) / (
                    self.metropolis_boltzmann * self.metropolis_temperature *
                    ((1 - self.metropolis_rate) ** self.iteration) + 1e-23)) > np.random.rand())

        # Probabilistic selection
        elif selector == 'probabilistic':
            selection_condition = bool((new <= old) or (np.random.rand() <= self.probability_selection))

        # All selection
        elif selector == 'all':
            selection_condition = True

        # None selection
        elif selector == 'none':
            selection_condition = False
        else:
            selection_condition = None
            raise PopulationError('Invalid selector!')

        return selection_condition


class PopulationError(Exception):
    """
    Simple PopulationError to manage exceptions.
    """
    pass
