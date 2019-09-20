#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 16:35:32 2019

@author: jkpvsz
"""
# Packages
import numpy as np
from population import Population 
from opteval import benchmark_func as bf
import matplotlib.pyplot as plt

#def run():
    # Problem definition
num_dimensions = 2
num_agents = 50
num_iterations = 100
problem = bf.Sphere(num_dimensions)
is_constrained = True
desired_fitness = 1E-6

problem.max_search_range = problem.max_search_range/100
problem.min_search_range = problem.min_search_range/100
    
    # Call optimisation method
#RandomSearch(problem, num_dimensions, num_agents, num_iterations, 
#                 desired_fitness, is_constrained)
    
#def RandomSearch(problem, num_dimensions = 2, num_agents = 30, 
#                     num_iterations = 100, desired_fitness = 1E-6, 
#                     is_constrained = True):
    
# Create population
pop = Population(problem,desired_fitness,num_agents,is_constrained)

# Initialise population with random elements uniformly distributed in space
pop.initialise_uniformly()

# -- plot population
plt.figure(1)
plt.ion()

plt.axis([-1.1,1.1,-1.1,1.1])
plt.plot(pop.positions[:,0],pop.positions[:,1],'ro')
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.pause(0.01)
plt.draw()

# Evaluate fitness values
pop.evaluate_fitness()

# Update population, global, etc
pop.update_population("all")
pop.update_particular("all")
pop.update_global("greedy")

# TODO initialise historical data

# Start optimisaton procedure
for iteration in range(1, num_iterations + 1):
    # Perform a perturbation
#    pop.random_walk(0.1)
#    pop.constricted_pso()
#    pop.inertial_pso()
    pop.levy_flight()
    
    # -- plot population
    plt.plot(pop.positions[:,0],pop.positions[:,1],'ro')
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.axis([-1.1,1.1,-1.1,1.1])
    plt.pause(0.01)
    plt.draw()
    
    # Evaluate fitness values
    pop.evaluate_fitness()

    # Update population, global
    pop.update_population("greedy")
    pop.update_global("greedy")
    
    # Print information per iteration
    if (iteration % 10) == 0:
        print("[",iteration,"] -> ",pop.get_state())
        
plt.ioff()
plt.show()
