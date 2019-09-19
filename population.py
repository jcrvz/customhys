# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 14:29:43 2019

@author: jkpvsz
"""
import numpy as np

class Population():

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
        
        # Initialise positions and fitness values
        self.positions = np.full((self.num_agents,self.num_dimensions),np.nan)
        self.fitness = np.full((self.num_agents,1),np.nan)
        
        # Initialise additional params (those corresponding to other algs)
        # -> Velocities for PSO-based search methods
        self.velocities = np.full((self.num_agents,self.num_dimensions),0) 
        
        # General fitness measurements
        self.global_best_fitness = float('inf')
        self.global_best_position = np.random.uniform(-1, 1, 
                                        (self.num_agents, self.num_dimensions))
    
        self.particular_best_fitness = self.fitness
        self.particular_best_position = self.positions
        
        self.previous_fitness = self.fitness
        self.previous_position = self.positions
        
        self.is_constrained = is_constrained
        
        # TODO Add capábility for dealing with topologies (neighbourhoods)
        # self.local_best_fitness = self.fitness
        # self.local_best_positions = self.positions
    
    # [E] Generate a string containing the current state of the population
    def get_state(self):
        print("x_best = ", self.global_best_position, " with f_best = ", 
              self.global_best_fitness)
    
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
    
    # [E] 2.1. Random Walk    
    def random_walk(self, scale = 0.01, *args):
        for agent in range(0, self.num_agents):
            self.positions[agent,:] += scale * np.random.uniform(-1, 1, 
                                          self.num_dimensions)
            if self.is_constrained: self.check_simple_constraints(agent)
    
    # [E] 2.2. Inertial PSO movement
    def inertial_pso(self, w = 0.7, phi_1 = 2.52, phi_2 = 2.53, *args):
        for agent in range(0, self.num_agents):
            r_1 = np.random.rand(self.num_dimensions)
            r_2 = np.random.rand(self.num_dimensions)
            self.velocities[agent,:] = w*self.velocities[agent,:] + \
                phi_1 * r_1 * (self.positions[agent,:] - \
                self.particular_best_position[agent,:]) + phi_2 * r_2 * \
                (self.positions[agent,:] - self.global_best_position)
            self.positions[agent,:] = self.positions[agent,:] + \
                self.velocities[agent,:]            
            if self.is_constrained: self.check_simple_constraints(agent)
    
    # [E] 2.3. Constricted PSO movement                
    def constricted_pso(self, chi = 0.7, phi_1 = 2.52, phi_2 = 2.53, *args):
        for agent in range(0, self.num_agents):
            r_1 = np.random.rand(self.num_dimensions)
            r_2 = np.random.rand(self.num_dimensions)
            self.velocities[agent,:] = chi*(self.velocities[agent,:] + \
                phi_1 * r_1 * (self.positions[agent,:] - \
                self.particular_best_position[agent,:]) + phi_2 * r_2 * \
                (self.positions[agent,:] - self.global_best_position))
            self.positions[agent,:] = self.positions[agent,:] + \
                self.velocities[agent,:]            
            if self.is_constrained: self.check_simple_constraints(agent)

    # -------------------------------------------------------------------------
    # 3. Basic methods
    # -------------------------------------------------------------------------
    
    # 3.1. Check simple contraints if self.is_constrained = True
    def check_simple_constraints(self, agent):
        
        low_check = self.positions[agent,:] < self.lower_boundaries
        if low_check.any():
            self.positions[agent, low_check] = self.lower_boundaries[low_check]
            self.velocities[agent, low_check] = 0.
            
        upp_check = self.positions[agent,:] > self.upper_boundaries
        if upp_check.any():
            self.positions[agent, upp_check] = self.upper_boundaries[upp_check]
            self.velocities[agent, upp_check] = 0.
        
    # 3.2. Rescale an agent from [-1,1] to [lower,upper] per dimension
    def rescale_back(self,agent):
        return ((self.upper_boundaries + self.lower_boundaries) + agent * \
                (self.upper_boundaries - self.lower_boundaries)) / 2

    # [E] 3.3. Evaluate population positions in the problem function
    def evaluate_fitness(self):
        for agent in range(0, self.num_agents):
            self.fitness[agent] = self.problem.get_func_val(
                    self.rescale_back(self.positions[agent,:]))
    
    # [E] 3.4. Update population positions according to a selection scheme
    def update_population(self, selection_method = "all"):
        for agent in range(0,self.num_agents):
            if locals()[selection_method+"_selection"](self.fitness[agent], 
                     self.previous_fitness[agent]):
                # if new positions are improved, then update past register ...
                self.previous_fitness[agent] = self.fitness[agent]
                self.previous_position[agent,:] = self.positions[agent,:]
            else:
                # ... otherwise,return to previous values
                self.fitness[agent] = self.previous_fitness[agent]
                self.positions[agent,:] = self.previous_position[agent,:]

    # [E] 3.5. Update particular positions acording to a selection scheme
    def update_particular(self, selection_method = "greedy"):        
        for agent in range(0,self.num_agents):
            if locals()[selection_method+"_selection"](self.fitness[agent], 
                     self.particular_best_fitness[agent]):
                self.particular_best_fitness[agent] = self.fitness[agent]
                self.particular_best_position[agent,:] =self.positions[agent,:]
    
    # 3.6. [E] Update global position according to a selection scheme
    def update_global(self, selection_method = "greedy"):        
        # Read current global best agent
        current_global_best_position = self.positions[self.fitness.argmin(),:]
        current_global_best_fitness = self.fitness.min()
        if locals()[selection_method+"_selection"](current_global_best_fitness, 
                 self.global_best_fitness):
            self.particular_best_position = current_global_best_position
            self.current_global_best_fitness = current_global_best_fitness 
    
    # -------------------------------------------------------------------------
    # 4. Selector methods
    # -------------------------------------------------------------------------
    
    def greedy_selection(new, old, *args):
        return new < old
    
    def none_selection(new, old, *args):
        return False
    
    def all_selection(new, old, *args):
        return True
    
    def metropolis_selection(new, old, T_0=1000, c=0.01, n=1, *args):
        # T_0 - initial temperature (T_0)
        # c - cooling rate (c)
        # n - current iteration (it | n)
        if new <= old: selection_conditon = True
        else:
            if np.math.exp(-(new - old)/(T_0*(1 - c)**n)) > np.random.rand(): 
                selection_conditon = True
            else: selection_conditon = False
        return selection_conditon 
        
    
    
    
        
        