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
#problem = bf.Rosenbrock(num_dimensions)
#problem = bf.Ackley(num_dimensions)
#problem = bf.Griewank(num_dimensions)

is_constrained = True
desired_fitness = 1e-6

# Find the proble function : objective function to minimise
problem_function = lambda x : problem.get_func_val(x)

# Define the problem domain
boundaries = (problem.max_search_range, problem.min_search_range)
    
# Create population
pop = Population(problem_function,boundaries,desired_fitness,num_agents,
                 is_constrained)

# Initialise population with random elements uniformly distributed in space
pop.initialise_uniformly()

# -- plot population
plt.figure(1)
plt.ion()

# Evaluate fitness values
pop.evaluate_fitness()

# Update population, global, etc
pop.update_population("all")
pop.update_particular("all")
pop.update_global("greedy")

# TODO initialise historical data

#pop.pso_self_confidence = 1.56  # <- inertial
#pop.pso_swarm_confidence = 1.56 # <-inertial 
#pop.pso_self_confidence = 2.25  # <- constraint
#pop.pso_swarm_confidence = 2.56 # <- constraint
#pop.pso_kappa = 1.0
#pop.pso_inertial = 0.6

#pop.cs_probability = 0.75

pop.metropolis_temperature = 100
pop.metropolis_rate = 1

pop.de_mutation_scheme = ("current-to-best",1)
pop.de_f = 1.0
pop.de_cr = 0.5

pop.spiral_radius = 0.95
pop.radius_span = 0.1

#pop.firefly_epsilon = "uniform"
#pop.firefly_gamma = 1.0
#pop.firefly_beta = 1.0
#pop.firefly_alpha = 0.8

#pop.cf_gravity = 0.001
#pop.cf_alpha = 0.001
#pop.cf_beta = 1.5
#pop.cf_time_interval = 1.0

pop.gs_gravity = 1
pop.gs_alpha = 2/100

plt.plot(pop.positions[:,0],pop.positions[:,1],'ro')
plt.plot(pop.global_best_position[0],pop.global_best_position[1],'bo')
plt.xlabel('x_1'); plt.ylabel('x_2'); plt.ylabel('x_3')
plt.title(str(pop.global_best_fitness)+" at "+str(0))
plt.axis([-1.1,1.1,-1.1,1.1])
plt.pause(0.01)
plt.draw()

# Start optimisaton procedure
for iteration in range(1, num_iterations + 1):
    # Perform a perturbation
#    pop.random_search()
#    pop.random_sample()
#    pop.rayleigh_flight()
#    pop.levy_flight()          # CS
#    pop.local_random_walk()    # CS
#    pop.mutation_de()          # DE
#    pop.firefly()              # FA
    
#    pop.constricted_pso()      # PSO
#    pop.inertial_pso()         # PSO     
    pop.spiral_dynamic()#0.90,22.5,0.2)       # D/S-SOA
#    pop.central_force()        # CFO    
#    pop.gravitational_search() # GSA    
    
#    pop.binomial_crossover_de() # DE
#    pop.exponential_crossover_de() # DE
        
    # Evaluate fitness values
    pop.evaluate_fitness()    

    # Update population, global
    pop.iteration = iteration
    pop.update_population("all")
    pop.update_global()
    
    # -- plot population
    plt.cla()
    plt.plot(pop.positions[:,0],pop.positions[:,1],'ro')
    plt.plot(pop.global_best_position[0],pop.global_best_position[1],'bo')
    plt.xlabel('x_1'); plt.ylabel('x_2'); plt.ylabel('x_3')
    plt.title(str(pop.global_best_fitness)+" at "+str(iteration))
    plt.axis([-1.1,1.1,-1.1,1.1])
    plt.pause(0.01)
    plt.draw()
    
#    print(pop.current_best_fitness,' ',pop.current_worst_fitness)
    
    # Print information per iteration
    if (iteration % 10) == 0:
        print("[",iteration,"] -> ",pop.get_state())
        
plt.ioff()
plt.show()
