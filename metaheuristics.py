#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 16:35:32 2019

@author: jkpvsz
"""
# Packages
from population import Population 
from opteval import benchmark_func as bf

def run():
    # Problem definition
    num_dimensions = 2
    num_agents = 30
    num_iterations = 100
    problem = bf.Sphere(num_dimensions)
    is_constrained = True
    desired_fitness = 1E-6
    
    # Call optimisation method
    RandomSearch(problem, num_dimensions, num_agents, num_iterations, 
                 desired_fitness, is_constrained)
    
def RandomSearch(problem, num_dimensions = 2, num_agents = 30, 
                     num_iterations = 100, desired_fitness = 1E-6, 
                     is_constrained = True):
    
    # Create population
    population = Population(problem,desired_fitness,num_agents,is_constrained)
    
    # Initialise population with random elements uniformly distributed in space
    population.initialise_uniformly()
    
    # Evaluate fitness values
    population.evaluate_fitness()
    
    # Update population, global, etc
    population.update_population("all")
    population.update_particular("greedy")
    population.update_global("greedy")
    
    # TODO initialise historical data
    
    # Start optimisaton procedure
    for iteration in range(1,num_dimensions + 1):
        # Perform a perturbation
        population.random_walk(0.01)
        
        # Evaluate fitness values
        population.evaluate_fitness()
    
        # Update population, global, etc
        population.update_population("all")
        population.update_particular("greedy")
        population.update_global("greedy")
        
        # Print information per iteration
        status = population.get_state()
        print(f"[{iteration}]\t-> {status}")
    