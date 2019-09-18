# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 14:29:43 2019

@author: jkpvsz
"""
import numpy as np

class Population():

    def __init__(self, problem, desired_fitness, num_agents):
        # Read problem (it must be from from opteval import benchmark_func as bf)
        self.problem = problem
        
        # Read desired fitness to achieve
        self.desired_fitness = desired_fitness
        
        # Read number of agents in population
        self.num_agents = num_agents
        
        # Read number of variables or dimension 
        self.num_dimensions = problem.variable_num
        
        # Read the upper and lower boundaries of search space
        self.lowerBoundaries = problem.min_search_range
        self.upperBoundaries = problem.max_search_range
        
        # Initialise positions and fitness values
        self.positions = np.full((self.num_agents,self.num_dimensions),np.nan)
        self.fitness = np.full((self.num_agents,1),np.nan)
        
        # Initialise additional parameters (those corresponding to other algorithms)
        # -> Velocities for PSO-based search methods
        self.velocities = np.full((self.num_agents,self.num_dimensions),np.nan) 
        
        # General fitness measurements
        self.global_best_fitness = float('inf')
        self.global_best_position = np.random.uniform(-1,1,(self.num_agents,self.num_dimensions))
    
        self.particular_best_fitness = self.fitness
        self.particular_best_position = self.positions
        
        self.previous_fitness = self.fitness
        self.previous_position = self.positions
        
        # TODO Add capábility for dealing with topologies (neighbourhoods)
        # self.local_best_fitness = self.fitness
        # self.local_best_positions = self.positions
        
    # Rescale an agent from [-1,1] to [lower,upper] per dimension
    def rescale_back(self,agent):
    # y = ((b - a) * x + (a + b)) / 2
        return ((self.upperBoundaries + self.lowerBoundaries) + agent * (self.upperBoundaries - self.lowerBoundaries)) / 2


    def evaluate_fitness(self):
        for agent in range(0, self.num_agents):
            self.fitness[agent] = self.problem(self.rescale_back(self.positions[agent,:]))
    
    def update_population(self,selection_method="greedy"):
        for agent in range(0,self.num_agents):
            if locals()[selection_method](self.fitness[agent], self.previous_fitness[agent]):
                # if new positions are improved, then update past register ...
                self.previous_fitness[agent] = self.fitness[agent]
                self.previous_position[agent,:] = self.positions[agent,:]
            else:
                # ... otherwise,return to previous values
                self.fitness[agent] = self.previous_fitness[agent]
                self.positions[agent,:] = self.previous_position[agent,:]

    def update_particular(self,selection_method="greedy"):        
        for agent in range(0,self.num_agents):
            if locals()[selection_method](self.fitness[agent], self.particular_best_fitness[agent]):
                self.particular_best_fitness[agent] = self.fitness[agent]
                self.particular_best_position[agent,:] = self.positions[agent,:]
    
#    def greedySelector(population, fitness, new_population, new_fitness):
#        # TODO improve by using np.ndarray.min(,axis) and np.ndarray.argmin(,axis)
#        for agent in range(0,population.shape[1]):
#            if new_fitness[agent] <= fitness[agent]:
#                fitness[agent] = new_fitness[agent]
#                population[:,agent] = new_population[:,agent]
#        return population, fitness
    
    def update_global(self,selection_method="greedy"):        
        # Read current global best agent
        current_global_best_position = self.positions[self.fitness.argmin(),:]
        current_global_best_fitness = self.fitness.min()
        
        # Check if global best is improved
        #if current_global_best_fitness < self.global_best_fitness:
        if locals()[selection_method](current_global_best_fitness, self.global_best_fitness):
            self.particular_best_position = current_global_best_position
            self.current_global_best_fitness = current_global_best_fitness 
    
    # Selection methods
    
    def greedy_selection(new,old,*args):
        return new < old
    
    def none_selection(new,old,*args):
        return False
    
    def metropolis_selection(new,old,*args):
        # args[0] - initial temperature (T_0)
        # args[1] - cooling rate (c)
        # args[2] - current iteration (it | n)
        #//print(f"T_0 = {args[0]}, c = {args[1]}, n = {args[2]},")
        energies_difference = new - old
        if energies_difference <= 0:
            selection_conditon = True
        else:
            current_temperature = args[0]*(1 - args[1])**args[2]
            boltzmann_probability = np.math.exp(-energies_difference/current_temperature)
            if boltzmann_probability > np.random.rand(): 
                selection_conditon = True
            else:
                selection_conditon = False
        return selection_conditon 
        
    
    
    
        
        