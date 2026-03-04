"""
This module contains the class Population.

Created on Tue Sep 17 14:29:43 2019

@author: Jorge Mario Cruz-Duarte (jcrvz.github.io), e-mail: j.m.cruzduarte@ieee.org
"""

from math import isfinite

import numpy as np
from scipy.stats.qmc import Halton, Sobol

__all__ = ["Population"]
__selectors__ = ["all", "greedy", "metropolis", "probabilistic"]


class Population:
    """
    This is the Population class, each object corresponds to a population of agents within a search space.
    """

    # Internal variables
    iteration = 0
    # rotation_matrix = []

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
            raise PopulationError("Lower and upper boundaries must have the same length")

        # Read the upper and lower boundaries of search space
        self.lower_boundaries = np.array(boundaries[0]) if isinstance(boundaries[0], list) else boundaries[0]
        self.upper_boundaries = np.array(boundaries[1]) if isinstance(boundaries[1], list) else boundaries[1]
        self.span_boundaries = self.upper_boundaries - self.lower_boundaries
        self.centre_boundaries = (self.upper_boundaries + self.lower_boundaries) / 2.0

        # Read number of agents in population
        assert isinstance(num_agents, int)
        self.num_agents = num_agents

        # Initialise positions and fitness values
        self._positions = np.full((self.num_agents, self.num_dimensions), np.nan)
        self.velocities = np.full((self.num_agents, self.num_dimensions), 0)
        self.fitness = np.full(self.num_agents, np.nan)

        # General fitness measurements
        self.global_best_position = np.full(self.num_dimensions, np.nan)
        self.global_best_fitness = float("inf")

        self.current_best_position = np.full(self.num_dimensions, np.nan)
        self.current_best_fitness = float("inf")
        self.current_worst_position = np.full(self.num_dimensions, np.nan)
        self.current_worst_fitness = -float("inf")

        self.particular_best_positions = np.full((self.num_agents, self.num_dimensions), np.nan)
        self.particular_best_fitness = np.full(self.num_agents, np.nan)

        self.previous_positions = np.full((self.num_agents, self.num_dimensions), np.nan)
        self.previous_velocities = np.full((self.num_agents, self.num_dimensions), np.nan)
        self.previous_fitness = np.full(self.num_agents, np.nan)

        self.backup_positions = np.full((self.num_agents, self.num_dimensions), np.nan)
        self.backup_velocities = np.full((self.num_agents, self.num_dimensions), np.nan)
        self.backup_fitness = np.full(self.num_agents, np.nan)
        self.backup_particular_best_positions = np.full((self.num_agents, self.num_dimensions), np.nan)
        self.backup_particular_best_fitness = np.full(self.num_agents, np.nan)

        self.is_constrained = is_constrained

        # TODO Add capability for dealing with topologies (neighbourhoods)
        # self.local_best_fitness = self.fitness
        # self.local_best_positions = self._positions

    # ===========
    # BASIC TOOLS
    # ===========

    @property
    def positions(self):
        return self._positions

    @positions.setter
    def positions(self, value):
        # Save the previous values
        self.previous_positions = np.copy(self._positions)
        self.previous_velocities = np.copy(self.velocities)
        self.previous_fitness = np.copy(self.fitness)

        # Remove the now current values
        self.velocities = np.full((self.num_agents, self.num_dimensions), 0)
        self.fitness = np.full(self.num_agents, np.nan)

        # Set the new values
        self._positions = value

    def get_state(self):
        """
        Return a string containing the current state of the population, i.e.,
            str = 'x_best = ARRAY, f_best = VALUE'

        :returns: str
        """
        return (
            "x_best = "
            + str(self.rescale_back(self.global_best_position))
            + ", f_best = "
            + str(self.global_best_fitness)
        )

    def get_positions(self):
        """
        Return the current population positions. Positions are represented in a matrix of size:
            ``positions.shape() = (num_agents, num_dimensions)``

        **NOTE:** The position is rescaled from the normalised search space, i.e., [-1, 1]^num_dimensions.

        :returns: numpy.ndarray
        """
        return np.tile(self.centre_boundaries, (self.num_agents, 1)) + self._positions * np.tile(
            self.span_boundaries / 2.0, (self.num_agents, 1)
        )

    def set_positions(self, positions):
        """
        Modify the current population positions. Positions are represented in a matrix of size:
            ``positions.shape() = (num_agents, num_dimensions)``

        Note: The position is rescaled to the original search space.

        :param numpy.ndarray positions:
            Population positions must have the size num_agents-by-num_dimensions array.

        :returns: numpy.ndarray
        """
        return (
            2.0
            * (positions - np.tile(self.centre_boundaries, (self.num_agents, 1)))
            / np.tile(self.span_boundaries, (self.num_agents, 1))
        )

    def revert_positions(self):
        """
        Revert the positions to the data in backup variables.
        """
        self.fitness = np.copy(self.backup_fitness)
        self._positions = np.copy(self.backup_positions)
        self.velocities = np.copy(self.backup_velocities)
        self.particular_best_fitness = np.copy(self.backup_particular_best_fitness)
        self.particular_best_positions = np.copy(self.backup_particular_best_positions)
        self.update_positions("global", "greedy")

    def update_positions(self, level: str = "population", selector: str | list[str] = "greedy"):
        # def update_positions(self, level: str ='population', selector: (str, list[str]) = 'greedy'):
        """
        Update the population positions according to the level and selection scheme.

        **NOTE:** When an operator is applied (from the operators' module), it automatically replaces new positions, so
        the logic of selectors is contrary as they commonly do.

        :param str level: optional
            Update level, it can be 'population' for the entire population, 'particular' for each agent_id (an
            historical performance), and 'global' for the current solution. The default is 'population'.
        :param str selector: optional
            Selection method. The selectors available are: 'greedy', 'probabilistic', 'metropolis', 'all', and 'none'.
            The default is 'all'.

        :returns: None.
        """
        # Check if the selector is a list

        # Update global positions, velocities and fitness
        if level == "global":
            if isinstance(selector, str):
                self.__selection_on_particular([selector] * self.num_agents)
                self.__selection_on_global(selector)
            else:
                raise PopulationError("Invalid global selector!")

        else:
            # Check if selector is a list and this has the same length as the number of agents
            if isinstance(selector, list):
                # Assert the length of the selector
                assert len(selector) == self.num_agents

            elif isinstance(selector, str):
                selector = [selector] * self.num_agents
            else:
                raise PopulationError("Invalid selector!")

            # Update population positions, velocities and fitness
            if level == "population":
                self.__selection_on_population(selector)

            # Update particular positions, velocities and fitness
            elif level == "particular":
                self.__selection_on_particular(selector)

            # Raise an error
            else:
                raise PopulationError("Invalid update level")

    def __selection_on_population(self, selector):
        # backup the previous position to prevent losses
        self.backup_fitness = np.copy(self.previous_fitness)
        self.backup_positions = np.copy(self.previous_positions)
        self.backup_velocities = np.copy(self.previous_velocities)

        for agent_id in range(self.num_agents):
            if self._selection(self.fitness[agent_id], self.previous_fitness[agent_id], selector[agent_id]):
                self.__update_best_and_worst()
                # if new positions are improved, then update the previous register
                # self.previous_fitness[agent_id] = np.copy(self.fitness[agent_id])
                # self.previous_positions[agent_id, :] = np.copy(self._positions[agent_id, :])
                # self.previous_velocities[agent_id, :] = np.copy(self.velocities[agent_id, :])

            else:
                # ... otherwise, return to previous values
                self.fitness[agent_id] = np.copy(self.previous_fitness[agent_id])
                self._positions[agent_id, :] = np.copy(self.previous_positions[agent_id, :])
                self.velocities[agent_id, :] = np.copy(self.previous_velocities[agent_id, :])

    def __update_best_and_worst(self):
        # Update the current best and worst positions (forced to greedy)
        self.current_best_position = np.copy(self._positions[self.fitness.argmin(), :])
        self.current_best_fitness = np.min(self.fitness)
        self.current_worst_position = np.copy(self._positions[self.fitness.argmax(), :])
        self.current_worst_fitness = np.min(self.fitness)

    def __selection_on_particular(self, selector):
        self.backup_particular_best_fitness = np.copy(self.particular_best_fitness)
        self.backup_particular_best_positions = np.copy(self.particular_best_positions)

        for agent_id in range(self.num_agents):
            if self._selection(
                self.fitness[agent_id], self.particular_best_fitness[agent_id], selector[agent_id]
            ) or not isfinite(self.particular_best_fitness[agent_id]):
                self.particular_best_fitness[agent_id] = np.copy(self.fitness[agent_id])
                self.particular_best_positions[agent_id, :] = np.copy(self._positions[agent_id, :])

    def __selection_on_global(self, selector="greedy"):
        # Read current global best agent_id
        candidate_position = np.copy(self.current_best_position)
        candidate_fitness = np.copy(self.current_best_fitness)
        if self._selection(candidate_fitness, self.global_best_fitness, selector) or not isfinite(candidate_fitness):
            self.global_best_position = np.copy(candidate_position)
            self.global_best_fitness = np.copy(candidate_fitness)

    def evaluate_fitness(self, problem_function):
        """
        Evaluate the population positions in the problem function.

        :param function problem_function:
            A function that maps a 1-by-D array of real values to a real value.

        :returns: None.
        """
        # Read problem, it must be a callable function
        assert callable(problem_function)

        # Check simple constraints before evaluate
        if self.is_constrained:
            self._check_simple_constraints()

        # Evaluate each agent in this function
        for agent in range(self.num_agents):
            self.fitness[agent] = problem_function(self.rescale_back(self._positions[agent, :]))

    # ==============
    # INITIALISATORS
    # ==============
    # TODO Add more initialisation operators like grid, boundary, etc.

    def initialise_positions(self, scheme="random"):
        """
        Initialise population by an initialisation scheme.

        :param str scheme: optional
            Initialisation scheme. Available schemes are:
            - 'random': Random uniform distribution in [-1,1] (default)
            - 'vertex': Vertices of nested hyper-cubes
            - 'lhs': Latin Hypercube Sampling
            - 'sobol': Sobol quasi-random sequences
            - 'halton': Halton quasi-random sequences
            - 'beta': Beta distribution
            - 'normal': Normal (Gaussian) distribution
            - 'lognormal': Lognormal distribution
            - 'exponential': Exponential distribution
            - 'rayleigh': Rayleigh distribution

        :returns: None.
        """
        scheme = "random" if scheme is None else str(scheme).strip().lower()

        if scheme == "vertex":
            self._positions = self._grid_matrix(self.num_dimensions, self.num_agents)
        elif scheme == "lhs":
            self._positions = self._lhs(self.num_dimensions, self.num_agents)
        elif scheme == "sobol":
            self._positions = self._sobol(self.num_dimensions, self.num_agents)
        elif scheme == "halton":
            self._positions = self._halton(self.num_dimensions, self.num_agents)
        elif scheme == "beta":
            self._positions = self._beta(self.num_dimensions, self.num_agents)
        elif scheme == "normal":
            self._positions = self._normal(self.num_dimensions, self.num_agents)
        elif scheme == "lognormal":
            self._positions = self._lognormal(self.num_dimensions, self.num_agents)
        elif scheme == "exponential":
            self._positions = self._exponential(self.num_dimensions, self.num_agents)
        elif scheme == "rayleigh":
            self._positions = self._rayleigh(self.num_dimensions, self.num_agents)
        else:
            # Default to random if scheme is not recognized
            self._positions = np.random.uniform(-1, 1, (self.num_agents, self.num_dimensions))

    # ================
    # INTERNAL METHODS
    # ================
    # Avoid using them outside

    @staticmethod
    def _grid_matrix(num_dimensions, num_agents):

        total_vertices = 2**num_dimensions

        basic_matrix = (
            2 * np.array([[int(x) for x in list(format(k, f"0{num_dimensions}b"))] for k in range(total_vertices)]) - 1
        )

        output_matrix = np.copy(basic_matrix)

        if num_agents > total_vertices:
            num_matrices = int(np.ceil((num_agents - total_vertices) / total_vertices)) + 1
            for k in range(1, num_matrices):
                k_matrix = (1 - k / num_matrices) * basic_matrix
                output_matrix = np.concatenate((output_matrix, k_matrix), axis=0)

        output_matrix = output_matrix[:num_agents, :]

        return output_matrix

    @staticmethod
    def _lhs(num_dimensions, num_agents):
        """
        Initialise population using Latin Hypercube Sampling (LHS).

        Ensures that each interval in each dimension is covered exactly once,
        providing stratified sampling across the search space.

        :param int num_dimensions: Number of dimensions
        :param int num_agents: Number of agents
        :returns: numpy.ndarray
        """
        segments = np.arange(num_agents) / num_agents
        segment_width = 1.0 / num_agents
        sample = np.zeros((num_agents, num_dimensions))

        for d in range(num_dimensions):
            perm = np.random.permutation(num_agents)
            for i in range(num_agents):
                offset = np.random.uniform(0, segment_width)
                sample[perm[i], d] = segments[i] + offset

        return sample * 2 - 1

    @staticmethod
    def _sobol(num_dimensions, num_agents):
        """
        Initialise population using Sobol quasi-random sequences.

        Sobol generates uniformly distributed points in the hypercube [0,1]^dim,
        providing better coverage than purely random sampling.

        Note: This method generates 2^m points where m = ceil(log2(num_agents)).
        If 2^m > num_agents, excess points are discarded.

        :param int num_dimensions: Number of dimensions
        :param int num_agents: Number of agents
        :returns: numpy.ndarray
        """
        m = int(np.ceil(np.log2(num_agents)))

        sobol = Sobol(d=num_dimensions, scramble=True)

        sample = sobol.random_base2(m)

        seq = sample[:num_agents]

        return seq * 2 - 1

    @staticmethod
    def _halton(num_dimensions, num_agents):
        """
        Initialise population using Halton quasi-random sequences.

        Halton generates uniformly distributed points in the hypercube [0,1]^dim,
        providing better coverage than purely random sampling.

        :param int num_dimensions: Number of dimensions
        :param int num_agents: Number of agents
        :returns: numpy.ndarray
        """
        halton = Halton(d=num_dimensions, scramble=True)

        sample = halton.random(n=num_agents)

        return sample * 2 - 1

    @staticmethod
    def _beta(num_dimensions, num_agents, a=0.5, b=0.5):
        """
        Initialise population using Beta distribution.

        Beta distribution generates points uniformly distributed in [0,1]^dim,
        with parameters controlling concentration towards edges or center.

        :param int num_dimensions: Number of dimensions
        :param int num_agents: Number of agents
        :param float a: Alpha parameter of Beta distribution (default: 0.5)
        :param float b: Beta parameter of Beta distribution (default: 0.5)
        :returns: numpy.ndarray
        """
        sample = np.random.beta(a, b, size=(num_agents, num_dimensions))

        return sample * 2 - 1

    @staticmethod
    def _normal(num_dimensions, num_agents, mean=0.5, std=0.25):
        """
        Initialise population using Normal (Gaussian) distribution.

        Normal distribution generates points concentrated around the center,
        with most points within 3 standard deviations.

        :param int num_dimensions: Number of dimensions
        :param int num_agents: Number of agents
        :param float mean: Mean of the normal distribution on [0,1] (default: 0.5)
        :param float std: Standard deviation on [0,1] (default: 0.25)
        :returns: numpy.ndarray
        """
        sample = np.random.normal(mean, std, size=(num_agents, num_dimensions))

        sample = np.clip(sample, 0, 1)

        return sample * 2 - 1

    @staticmethod
    def _lognormal(num_dimensions, num_agents, mean=0.0, sigma=0.5):
        """
        Initialise population using Lognormal distribution.

        Lognormal distribution is suitable for generating points with
        positive skewness and concentration towards smaller values.

        :param int num_dimensions: Number of dimensions
        :param int num_agents: Number of agents
        :param float mean: Mean of underlying normal distribution (default: 0.0)
        :param float sigma: Standard deviation of underlying normal (default: 0.5)
        :returns: numpy.ndarray
        """
        sample = np.random.lognormal(mean, sigma, size=(num_agents, num_dimensions))

        sample = np.clip(sample, 0, 1)

        return sample * 2 - 1

    @staticmethod
    def _exponential(num_dimensions, num_agents, scale=0.5):
        """
        Initialise population using Exponential distribution.

        Exponential distribution generates points with exponential decay,
        concentrated towards zero.

        :param int num_dimensions: Number of dimensions
        :param int num_agents: Number of agents
        :param float scale: Scale parameter (default: 0.5)
        :returns: numpy.ndarray
        """
        sample = np.random.exponential(scale, size=(num_agents, num_dimensions))

        sample = np.clip(sample, 0, 1)

        return sample * 2 - 1

    @staticmethod
    def _rayleigh(num_dimensions, num_agents, scale=0.4):
        """
        Initialise population using Rayleigh distribution.

        Rayleigh distribution generates points with concentration
        away from zero, suitable for magnitude-based initialization.

        :param int num_dimensions: Number of dimensions
        :param int num_agents: Number of agents
        :param float scale: Scale parameter (default: 0.4)
        :returns: numpy.ndarray
        """
        sample = np.random.rayleigh(scale, size=(num_agents, num_dimensions))

        sample = np.clip(sample, 0, 1)

        return sample * 2 - 1

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
        if np.any(np.isnan(self._positions)):
            np.nan_to_num(self._positions, copy=False, nan=1.0, posinf=1.0, neginf=-1.0)

        # Check if agents are beyond lower boundaries
        low_check = np.less(self._positions, -1.0)
        if np.any(low_check):
            # Fix them
            self._positions[low_check] = -1.0
            self.velocities[low_check] = 0.0

        # Check if agents are beyond upper boundaries
        upp_check = np.greater(self._positions, 1.0)
        if np.any(upp_check):
            # Fix them
            self._positions[upp_check] = 1.0
            self.velocities[upp_check] = 0.0

    def rescale_back(self, position):
        """
        Rescale an agent position from [-1.0, 1.0] to the original search space boundaries per dimension.

        :param numpy.ndarray position:
            A position given by an array of 1-by-D with elements between [-1, 1].

        :returns: ndarray
        """
        return self.centre_boundaries + position * (self.span_boundaries / 2)

    def _selection(self, new, old, selector="greedy"):
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
        if not isfinite(old):
            return True

        if selector == "greedy":
            return new <= old

        # Metropolis selection
        elif selector == "metropolis":
            if new <= old:
                selection_condition = True
            else:
                selection_condition = bool(
                    np.exp(
                        -(new - old)
                        / (
                            self.metropolis_boltzmann
                            * self.metropolis_temperature
                            * ((1 - self.metropolis_rate) ** self.iteration)
                            + 1e-23
                        )
                    )
                    > np.random.rand()
                )
            return selection_condition

        # Probabilistic selection
        elif selector == "probabilistic":
            return bool((new <= old) or (np.random.rand() <= self.probability_selection))

        # All selection
        elif selector in ["all", "direct"]:
            return True

        # None selection
        elif selector == "none":
            return False
        else:
            raise PopulationError("Invalid selector!")
            return None


class PopulationError(Exception):
    """
    Simple PopulationError to manage exceptions.
    """

    pass
