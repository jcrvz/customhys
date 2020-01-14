# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 14:12:45 2020

@author: L03130342
"""

# Packages
import numpy as np
import population as Pop 
import operators as Ope
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
# problem = bf.Griewank(num_dimensions)

problem_function = lambda x : problem.get_func_val(x)
is_constrained = True

problem.max_search_range = problem.max_search_range/10
problem.min_search_range = problem.min_search_range/10
    
boundaries = (problem.min_search_range, problem.max_search_range)
    
# Create population
pop = Pop.Population(boundaries, num_agents, is_constrained)

# Initialise population with random elements uniformly distributed in space
pop.initialise_positions()

# -- plot population
plt.figure(1)
plt.ion()

# Evaluate fitness values
pop.evaluate_fitness(problem_function)

# Update population, global, etc
pop.update_positions('population', 'all')
pop.update_positions('global', 'all')

print("[0] -> ", pop.get_state())

plt.plot(pop.positions[:,0], pop.positions[:,1],'ro')
plt.plot(pop.global_best_position[0], pop.global_best_position[1],'bo')
plt.xlabel('x_1'); plt.ylabel('x_2'); plt.ylabel('x_3')
plt.title(str(pop.global_best_fitness)+" at "+str(0))
plt.axis([-1.1,1.1,-1.1,1.1])
plt.pause(0.01)
plt.draw()

Ope.random_sample(pop)

# Evaluate fitness values
pop.evaluate_fitness(problem_function)    

# Update population, global
# pop.update_positions('population', 'none')
pop.update_positions('global', 'greedy')

plt.plot(pop.positions[:,0], pop.positions[:,1],'ro')
plt.plot(pop.global_best_position[0], pop.global_best_position[1],'bo')
plt.xlabel('x_1'); plt.ylabel('x_2'); plt.ylabel('x_3')
plt.title(str(pop.global_best_fitness)+" at "+str(0))
plt.axis([-1.1,1.1,-1.1,1.1])
plt.pause(0.01)
plt.draw()

# Start optimisaton procedure
for iteration in range(1, num_iterations + 1):
    # Apply an operator
    # Ope.spiral_dynamic(pop)
    # Ope.central_force_dynamic(pop)
    # Ope.gravitational_search(pop)
    Ope.swarm_dynamic(pop)
    # Ope.genetic_mutation(pop)
    # Ope.random_sample(pop)
    # Ope.local_random_walk(pop)
    # Ope.firefly_dynamic(pop)
    # Ope.random_search(pop)
    # Ope.rayleigh_flight(pop)
    # Ope.levy_flight(pop)
    # Ope.differential_mutation(pop)
    # Ope.differential_crossover(pop)
    # Ope.genetic_crossover(pop)
    
    # Evaluate fitness values
    pop.evaluate_fitness(problem_function)    

    # Update population, global
    pop.iteration = iteration
    # pop.update_positions('population', 'metropolis')
    pop.update_positions('global', 'greedy')
    
    # -- plot population
    plt.cla()
    plt.plot(pop.positions[:,0],pop.positions[:,1],'ro')
    plt.plot(pop.global_best_position[0],pop.global_best_position[1],'bo')
    plt.xlabel('x_1'); plt.ylabel('x_2'); plt.ylabel('x_3')
    plt.title(str(pop.global_best_fitness)+" at "+str(iteration))
    plt.axis([-1.1,1.1,-1.1,1.1])
    plt.pause(0.01)
    plt.draw()
    
    # Print information per iteration
    if (iteration % 10) == 0:
        print("[",iteration,"] -> ",pop.get_state())
        
plt.ioff()
plt.show()

