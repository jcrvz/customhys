# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 10:04:15 2020

@author: L03130342
"""
# from Population class

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
            
    # Genetic Algorithm (GA): Selection strategies
    # -----------------------------------------------------------------------
    # -> Natural selection to obtain the mating pool
    def _ga_natural_selection(self, num_mates):
        # Sort population according to its fitness values
        sorted_indices = np.argsort(self.fitness)

        # Return indices corresponding mating pool
        return sorted_indices[:num_mates]

    # Genetic Algorithm (GA): Pairing strategies
    #

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

    # -> Random pairing
    def _ga_random_pairing(self, mating_pool, num_couples, *args):
        # Return two random indices from mating pool
        return mating_pool[np.random.randint(mating_pool.size,
                                             size=(2, num_couples))]
    
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
    
couple_indices = getattr(self, "_ga_" + pairing + "_pairing")(
                    mating_pool_indices, num_couples, int(tournament_size),
                    float(tournament_probability)/100)

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

 # Two-Point Crossover
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
    
# Before: interial_pso
def inertial_swarm(pop, inertial=0.7, self_conf=1.54, swarm_conf=1.56):
    """
    Performs a swarm movement by using the inertial version of Particle
    Swarm Optimisation (PSO).

    Parameters
    ----------
    pop : population
        It is a population object.
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
    _check_parameter('inertial')
    _check_parameter('self_conf', (0.0, 10.0))
    _check_parameter('swarm_conf', (0.0, 10.0))

    # Determine random numbers
    r_1 = self_conf * np.random.rand(pop.num_agents, pop.num_dimensions)
    r_2 = swarm_conf * np.random.rand(pop.num_agents, pop.num_dimensions)

    # Find new velocities
    pop.velocities = inertial * pop.velocities + r_1 * (
            pop.particular_best_positions - pop.positions) + \
        r_2 * (np.tile(pop.global_best_position, (pop.num_agents, 1)) -
               pop.positions)

    # Move each agent using velocity's information
    pop.positions += pop.velocities

    # Check constraints
    if pop.is_constrained:
        pop.__check_simple_constraints()


# Before: constriction_pso
def constriction_swarm(pop, kappa=1.0, self_conf=2.54, swarm_conf=2.56):
    """
    Performs a swarm movement by using the constricted version of Particle
    Swarm Optimisation (PSO).

    Parameters
    ----------
    pop : population
        It is a population object.
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
    _check_parameter('kappa')
    _check_parameter('self_conf', (0.0, 10.0))
    _check_parameter('swarm_conf', (0.0, 10.0))

    # Find the constriction factor chi using phi
    phi = self_conf + swarm_conf
    if phi > 4:
        chi = 2 * kappa / np.abs(2 - phi - np.sqrt(phi ** 2 - 4 * phi))
    else:
        chi = np.sqrt(kappa)

    # Determine random numbers
    r_1 = self_conf * np.random.rand(pop.num_agents, pop.num_dimensions)
    r_2 = swarm_conf * np.random.rand(pop.num_agents, pop.num_dimensions)

    # Find new velocities
    pop.velocities = chi * (pop.velocities + r_1 * (
        pop.particular_best_positions - pop.positions) +
        r_2 * (np.tile(pop.global_best_position, (pop.num_agents, 1)) -
               pop.positions))

    # Move each agent using velocity's information
    pop.positions += pop.velocities

    # Check constraints
    if pop.is_constrained:
        pop.__check_simple_constraints()
        
        
# before: binomial_crossover_de
def binomial_crossover(pop, crossover_rate=0.5):
    """
    Performs the binomial crossover from Differential Evolution (DE).

    Parameters
    ----------
    pop : population
        It is a population object.
    crossover_rate : float, optional
        Probability factor to perform the crossover. The default is 0.5.

    Returns
    -------
    None.

    """
    # Check the scale and beta value
    _check_parameter('crossover_rate')

    # Define indices
    indices = np.tile(np.arange(pop.num_dimensions), (pop.num_agents, 1))

    # Permute indices per dimension
    rand_indices = np.vectorize(np.random.permutation,
                                signature='(n)->(n)')(indices)

    # Calculate the NOT condition (because positions were already updated!)
    condition = np.logical_not((indices == rand_indices) | (
        np.random.rand(pop.num_agents, pop.num_dimensions) <=
        crossover_rate))

    # Reverse the ones to their previous positions
    pop.positions[condition] = pop.previous_positions[condition]

    # Check constraints
    if pop.is_constrained:
        pop.__check_simple_constraints()
        
# Before: exponentinal_crossover_de
def exponential_crossover(pop, crossover_rate=0.5):
    """
    Performs the exponential crossover from Differential Evolution (DE)

    Parameters
    ----------
    pop : population
        It is a population object.
    crossover_rate : float, optional
        Probability factor to perform the crossover. The default is 0.5.

    Returns
    -------
    None.

    """
    # Check the scale and beta value
    _check_parameter('crossover_rate')

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

    # Check constraints
    if pop.is_constrained:
        pop.__check_simple_constraints()