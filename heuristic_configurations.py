# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:09:11 2019

@author: L03130342
"""
import numpy as np

# List of several possible simple-heuristics and their default parameters
simple_heuristics = [
        # ********************** LOCAL RANDOM WALK ****************************
        ("local_random_walk", dict(probability=0.75, scale=1.00), "greedy"),    # 1
        ('random_search', dict(scale=0.01), "greedy"),                          # 2
        ("random_sample", dict(), "greedy"),                                    # 3
        ("rayleigh_flight", dict(scale=0.01), "greedy"),                        # 4
        ("levy_flight", dict(scale=1.0, beta=1.5), "greedy"),                   # 5
        ("mutation_de", dict(expression="current-to-best", num_rands=1,         # 6
                             factor=1.0), "greedy"),
        ('binomial_crossover_de', dict(crossover_rate=0.5), "greedy"),          # 7
        ("exponential_crossover_de", dict(crossover_rate=0.5), "greedy"),       # 8
        ("firefly", dict(epsilon="uniform", alpha=0.8, beta=1.0, gamma=1.0),    # 9
         "greedy"),     
        ("inertial_pso", dict(inertial=0.7, self_conf=1.54, swarm_conf=1.56),   # 10
          "all"),          
        ("constriction_pso", dict(kappa=1.0, self_conf=2.54, swarm_conf=2.56),  # 11
         "all"),          
        ("gravitational_search", dict(gravity=1.0, alpha=0.02, epsilon=1e-23),  # 12
         "all"),       
        ("central_force", dict(gravity=1, alpha=0.02, beta=1.5, dt=1.0), "all"),# 13
        ("spiral_dynamic", dict(radius=0.9, angle=22.5, sigma=0.1), "all"),     # 14
        ("ga_mutation", dict(elite_rate=0.0, mutation_rate=0.2,                 # 15
                             distribution="uniform", sigma=1.0), "all"),        
        ("ga_crossover", dict(pairing="cost", crossover="single",               # 16
                              mating_pool_factor=0.1), "all")
    ]

# number of possible valuer for continuous parameters
num_vals = 5

# list of possible parameters for each simple-heuristic in simple_heuristic
possible_parameters = [
        # n - (heuristic name, par1, par2, ..., recommended selector)
        ("local_random_walk", dict(probability = np.linspace(0.0,1.0,num_vals), 
                                   scale = np.linspace(0.0,1.0,num_vals)), "greedy"),
        ('random_search', dict(scale=np.linspace(0.0,1.0,num_vals)), "greedy"),  
        ("random_sample", dict(), "greedy"),
        ("rayleigh_flight", dict(scale=np.linspace(0.0,1.0,num_vals)), "greedy"),
        ("levy_flight", dict(scale=np.linspace(0.0,1.0,num_vals), beta=[1.5]), "greedy"),
        ("mutation_de", dict(expression=["rand", "best", "current", "current-to-best", 
                                         "rand-to-best", "rand-to-bestandcurrent"], 
                             num_rands=[1,2,3], 
                             factor=np.linspace(0.0,2.0,num_vals)), "greedy"),
        ('binomial_crossover_de', dict(
                         crossover_rate=np.linspace(0.0,1.0,num_vals)), "greedy"),
        ("exponential_crossover_de", dict(
                         crossover_rate=np.linspace(0.0,1.0,num_vals)), "greedy"),
        ("firefly", dict(epsilon=["uniform", "gaussian"], 
                         alpha=np.linspace(0.0,1.0,num_vals), 
                         beta=[1.0], 
                         gamma=np.linspace(1.0,100.0,num_vals)), "greedy"),
        ("inertial_pso", dict(inertial=np.linspace(0.0,1.0,num_vals), 
                              self_conf=np.linspace(0.0,5.0,num_vals), 
                              swarm_conf=np.linspace(0.0,5.0,num_vals)), "all"), 
        ("constriction_pso", dict(kappa=np.linspace(0.0,1.0,num_vals), 
                                  self_conf=np.linspace(0.0,5.0,num_vals), 
                                  swarm_conf=np.linspace(0.0,5.0,num_vals)), "all"),
        ("gravitational_search", dict(gravity=np.linspace(0.0,1.0,num_vals), 
                                      alpha=np.linspace(0.0,1.0,num_vals), 
                                      epsilon=[1e-23]), "all"),
        ("central_force", dict(gravity=np.linspace(0.0,1.0,num_vals), 
                               alpha=np.linspace(0.0,1.0,num_vals), 
                               beta=np.linspace(0.0,3.0,num_vals), 
                               dt=[1.0]), "all"),   
        ("spiral_dynamic", dict(radius=np.linspace(0.0,1.0,num_vals), 
                                angle=np.linspace(0.0,1.0,num_vals), 
                                sigma=np.linspace(0.0,1.0,num_vals)), "all"),
        ("ga_mutation", dict(elite_rate=np.linspace(0.0,1.0,num_vals), 
                             mutation_rate=np.linspace(0.0,1.0,num_vals), 
                             distribution=["uniform", "gaussian"], 
                             sigma=np.linspace(0.0,1.0,num_vals)), "all"),
        ("ga_crossover", dict(pairing=["even-odd", "rank", "cost", 
                                       "tournament_2_100", "tournament_2_75", 
                                       "tournament_3_100", "tournament_3_75"],
                             crossover=["single", "two", "uniform", "blend", 
                                        "linear_0.5_0.5", "linear_1.5_0.5", 
                                        "linear_0.5_1.5", "linear_1.5_1.5", 
                                        "linear_-0.5_0.5", "linear_0.5_-0.5"], 
                              mating_pool_factor=np.linspace(0.0,1.0,num_vals)), "all")
        ]

def generate_heuristics(heuristics = possible_parameters):
    # Counters: [classes, methods]
    total_counters = [0, 0]
    
    # Initialise the collection of simple heuristics
    file = open('operators_collection.txt', 'w')
        
    # For each simple heuristic, read their parameters and values
    for operator, parameters, selector in heuristics:
        # Update the total classes counter
        total_counters[0] += 1
        
        # Read the number of parameters and how many values have each one
        num_parameters = len(parameters)
        if num_parameters > 0:
            # Read the name and possible values of parameters
            par_names = list(parameters.keys())
            par_values = list(parameters.values())
            
            # Find the number of values for each parameter
            par_num_values = [np.size(x) for x in par_values]
            
            # Determine the number of combinations
            num_combinations = np.prod(par_num_values)
            
            # Create the table of all possible combinations (index per parameter)
            indices = [x.flatten() for x in np.meshgrid(
                *list(map(lambda x: np.arange(x), par_num_values)))]
            
            # For each combination, create a single dictionary which corresponds 
            # to a simple search operator
            for combi in range(num_combinations):
                list_tuples = [(par_names[k], par_values[k][indices[k][combi]]) 
                    for k in range(num_parameters)]
                simple_par_combination = dict(list_tuples)
                file.write(f"('{operator}', {simple_par_combination}, " + 
                      f"'{selector}')\n")
        else:
            num_combinations = 0
            
        # Update the total combination counter
        total_counters[1] += num_combinations
            
        print(f"{operator}: parameters={num_parameters}, " + 
              f"combinations:{num_combinations}")
    
    file.close()
    print("-" * 50 + "--\nTOTAL: classes=%d, operators=%d" % 
          tuple(total_counters))        
           