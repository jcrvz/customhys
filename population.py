# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 14:29:43 2019

@author: jkpvsz
"""
import numpy as np

__operators__ = ['random_search', 'random_sample', 'rayleigh_flight', 'inertial_pso',
                 'constricted_pso','levy_flight']
__selection__ = ['all','none','greedy','metropolis']

class Population():
    
    # Internal variables
    iteration = 1
    
    # Parameters per selection method    
    metropolis_temperature = 1000
    metropolis_rate = 0.01
    
    # Parameters per perturbation operator
    pso_self_confidence = 2.25
    pso_swarm_confidence = 2.52
    pso_kappa = 1.0
    pso_inertial = 0.8
    
    levy_alpha = 1.0
    levy_beta = 1.5
    
    random_scale = 0.01

    def __init__(self, problem, desired_fitness = 1E-6, num_agents = 30, 
                 is_constrained = True):        
        # Read problem (from opteval import benchmark_func as bf)
        self.problem = problem
        
        # Read desired fitness to achieve
        self.desired_fitness = desired_fitness
        
        # Read number of agents in population
        self.num_agents = num_agents
        
        # Read number of variables or dimension 
        self.num_dimensions = problem.variable_num
        
        # Read the upper and lower boundaries of search space
        self.lower_boundaries = problem.min_search_range
        self.upper_boundaries = problem.max_search_range
        self.span_boundaries = np.abs(self.upper_boundaries -
                self.lower_boundaries)
        
        # Initialise positions and fitness values
        self.positions = np.full((self.num_agents, self.num_dimensions),np.nan)
        self.fitness = np.full(self.num_agents, np.nan)
        
        # Initialise additional params (those corresponding to other algs)
        # -> Velocities for PSO-based search methods
#        self.velocities = np.random.uniform(-self.span_boundaries, 
#                self.span_boundaries, (self.num_agents, self.num_dimensions))
        self.velocities = np.full((self.num_agents, self.num_dimensions), 0)
        
        # General fitness measurements        
        self.global_best_position = np.full(self.num_dimensions, np.nan)
        self.global_best_fitness = float('inf')
    
        self.particular_best_positions = np.full(
                (self.num_agents,self.num_dimensions), np.nan)
        self.particular_best_fitness = np.full(self.num_agents, np.nan)
        
        self.previous_positions = np.full((self.num_agents, 
                                           self.num_dimensions), np.nan)
        self.previous_fitness = np.full(self.num_agents,np.nan)
        
        self.previous_velocities = np.full((self.num_agents, 
                                            self.num_dimensions), np.nan)
        
        self.is_constrained = is_constrained
        
        # TODO Add capábility for dealing with topologies (neighbourhoods)
        # self.local_best_fitness = self.fitness
        # self.local_best_positions = self.positions
    
    # [E] Generate a string containing the current state of the population
    def get_state(self):
        return ("x_best = "+str(self.rescale_back(self.global_best_position))+
                ", with f_best = "+str(self.global_best_fitness))
    
    # -------------------------------------------------------------------------
    # 1. Initialisators
    # -------------------------------------------------------------------------
    
    # [E] 1.1. Initialise population using a random uniform distribution in [-1,1]]
    def initialise_uniformly(self):
        for agent in range(0, self.num_agents):
            self.positions[agent] = np.random.uniform(-1, 1, 
                                      self.num_dimensions)   
    
    #TODO Add more initialisation operators like grid, boundary, etc.
    
    # -------------------------------------------------------------------------
    # 2. Perturbators
    # -------------------------------------------------------------------------
    
    # [E] 2.1. Random sample  
    def random_sample(self):
        self.positions = np.random.uniform(-1,1,(self.num_agents, 
            self.num_dimensions))
        if self.is_constrained: self.check_simple_constraints()
    
    # [E] 2.1. Random Walk    
    def random_search(self):
        self.positions += self.random_scale * np.random.uniform(-1,1,
            (self.num_agents, self.num_dimensions))
        if self.is_constrained: self.check_simple_constraints()    
    
    def rayleigh_flight(self):
        self.positions += self.random_scale * np.random.standard_normal(
            (self.num_agents, self.num_dimensions))
        if self.is_constrained: self.check_simple_constraints()
    
    # [E] 2.2. Inertial PSO movement
    def inertial_pso(self):        
        r_1 = self.pso_self_confidence * \
                np.random.uniform(0,1,(self.num_agents, self.num_dimensions))
        r_2 = self.pso_swarm_confidence * \
                np.random.uniform(0,1,(self.num_agents, self.num_dimensions))
        
        self.velocities = self.pso_inertial * self.velocities + r_1 * \
                (self.particular_best_positions - self.positions) + r_2 * \
                (np.tile(self.global_best_position, (self.num_agents, 1)) - \
                 self.positions)
        self.positions += self.velocities
        if self.is_constrained: self.check_simple_constraints()
    
    # [E] 2.3. Constricted PSO movement                
    def constricted_pso(self):
        phi = self.pso_self_confidence + self.pso_swarm_confidence
        chi = 2 * self.pso_kappa / np.abs(2 - phi - np.sqrt(phi**2 - 4*phi))
        r_1 = self.pso_self_confidence * \
                np.random.uniform(0,1,(self.num_agents, self.num_dimensions))
        r_2 = self.pso_swarm_confidence * \
                np.random.uniform(0,1,(self.num_agents, self.num_dimensions))
        
        self.velocities = chi *(self.velocities + r_1 * \
                (self.particular_best_positions - self.positions) + r_2 * \
                (np.tile(self.global_best_position, (self.num_agents, 1)) - \
                 self.positions))
        self.positions += self.velocities
        if self.is_constrained: self.check_simple_constraints()
        
    # [E] 2.4. Lévy flight from Cuckoo Search
    def levy_flight(self):
        # Calculate std dev of u (Mantegna's algorithm)
        sigma = ((np.math.gamma(1 + self.levy_beta) * np.sin(np.pi*
            self.levy_beta/2)) / (np.math.gamma((1 + self.levy_beta)/2)*\
            self.levy_beta*(2**((self.levy_beta - 1)/2))))** (1/self.levy_beta)
        
        x = sigma * np.random.standard_normal((self.num_agents, 
                self.num_dimensions))
        y = np.abs(np.random.standard_normal((self.num_agents, 
                self.num_dimensions)))
        z = np.random.standard_normal((self.num_agents, self.num_dimensions))
        
        levy = x / (y ** (1/self.levy_beta))
        
        self.positions += self.levy_alpha * z * levy * (self.positions - \
                np.tile(self.global_best_position, (self.num_agents, 1)))
        
                
    # -------------------------------------------------------------------------
    # 3. Basic methods
    # -------------------------------------------------------------------------
    
    # 3.1. Check simple contraints if self.is_constrained = True
    def check_simple_constraints(self):
        
        low_check = self.positions < -1.
        if low_check.any():
            self.positions[low_check] = -1.
            self.velocities[low_check] = 0.
            
        upp_check = self.positions > 1.
        if upp_check.any():
            self.positions[upp_check] = 1.
            self.velocities[upp_check] = 0.
        
    # 3.2. Rescale an agent from [-1,1] to [lower,upper] per dimension
    def rescale_back(self, position):
        return ((self.upper_boundaries + self.lower_boundaries) + position * \
                (self.upper_boundaries - self.lower_boundaries)) / 2

    # [E] 3.3. Evaluate population positions in the problem function
    def evaluate_fitness(self):
        for agent in range(0, self.num_agents):
            self.fitness[agent] = self.problem.get_func_val(
                    self.rescale_back(self.positions[agent,:]))
    
    # [E] 3.4. Update population positions according to a selection scheme
    def update_population(self, selection_method = "all"):
        for agent in range(0,self.num_agents):
            if getattr(self,selection_method+"_selection")(self.fitness[agent], 
                     self.previous_fitness[agent]):
                # if new positions are improved, then update past register ...
                self.previous_fitness[agent] = self.fitness[agent]
                self.previous_positions[agent,:] = self.positions[agent,:]
                self.previous_velocities[agent,:] = self.velocities[agent,:]
            else:
                # ... otherwise,return to previous values
                self.fitness[agent] = self.previous_fitness[agent]
                self.positions[agent,:] = self.previous_positions[agent,:]
                self.velocities[agent,:] = self.previous_velocities[agent,:]

    # [E] 3.5. Update particular positions acording to a selection scheme
    def update_particular(self, selection_method = "greedy"):        
        for agent in range(0, self.num_agents):            
            if getattr(self,selection_method+"_selection")(self.fitness[agent], 
                     self.particular_best_fitness[agent]):
                self.particular_best_fitness[agent] = self.fitness[agent]
                self.particular_best_positions[agent,:]=self.positions[agent,:]
    
    # 3.6. [E] Update global position according to a selection scheme
    def update_global(self, selection_method = "greedy"):       
        self.update_particular(selection_method)
        # Read current global best agent
        candidate_position = self.particular_best_positions[\
                self.particular_best_fitness.argmin(),:]
        candidate_fitness = self.particular_best_fitness.min()
        if getattr(self,selection_method+"_selection")(candidate_fitness, 
                  self.global_best_fitness) or np.isinf(candidate_fitness):
            self.global_best_position = candidate_position
            self.global_best_fitness = candidate_fitness 
    
    # -------------------------------------------------------------------------
    # 4. Selector methods
    # -------------------------------------------------------------------------

    def greedy_selection(self, new, old):
        return new <= old
    
    def none_selection(self, *args):
        return False
    
    def all_selection(self, *args):
        return True
    
    def metropolis_selection(self, new, old):
        # It depends of metropolis_temperature, metropolis_rate, and iteration
        if new <= old: selection_conditon = True
        else:
            if np.math.exp(-(new - old)/(self.metropolis_temperature*\
                (1 - self.metropolis_rate)**self.iteration))> np.random.rand(): 
                selection_conditon = True
            else: selection_conditon = False
        return selection_conditon 
    
    
        
        