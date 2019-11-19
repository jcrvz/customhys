# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:09:11 2019

@author: L03130342
"""

# List of several possible simple-heuristics
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

# list of possible parameters for each simple-heuristic in simple_heuristic

possible_parameters = [
        # n - (heuristic name, par1, par2, ..., recommended selector)
        ("local_random_walk", dict(probability=[0.0, 1.0], scale=[0.0, 1.0]), "greedy"),
        ('random_search', dict(scale=[0.0, 1.0]), "greedy"),  
        ("random_sample", dict(), "greedy"),
        ("rayleigh_flight", dict(scale=[0.0, 1.0]), "greedy"),
        ("levy_flight", dict(scale=[0.0, 1.0], beta=1.5), "greedy"),
        ("mutation_de", dict(expression=["rand", "best", "current", "current-to-best", 
                                         "rand-to-best", "rand-to-bestandcurrent"], 
                             num_rands=[1,3], factor=[0.0, 2.0]), "greedy"),
        ('binomial_crossover_de', dict(crossover_rate=[0.0, 1.0]), "greedy"),
        ("exponential_crossover_de", dict(crossover_rate=[0.0, 1.0]), "greedy"),
        ("firefly", dict(epsilon=["uniform", "gaussian"], alpha=[0.0, 1.0], beta=1.0, 
                         gamma=[1.0, 100.0]), "greedy"),
        ("inertial_pso", dict(inertial=[0.0, 1.0], self_conf=[0.0, 5.0], 
                              swarm_conf=[0.0, 5.0]), "all"), 
        ("constriction_pso", dict(kappa=[0.0, 1.0], self_conf=[0.0, 5.0], 
                                  swarm_conf=[0.0, 5.0]), "all"),
        ("gravitational_search", dict(gravity=[0.0, 1.0], alpha=[0.0, 1.0], 
                                      epsilon=1e-23), "all"),
        ("central_force", dict(gravity=[0.0, 1.0], alpha=[0.0, 1.0], 
                               beta=[0.0, 1.0], dt=1.0), "all"),   
        ("spiral_dynamic", dict(radius=[0.0, 1.0], angle=[0.0, 180], 
                                sigma=[0.0, 1.0]), "all"),
        ("ga_mutation", dict(elite_rate=[0.0, 1.0], mutation_rate=[0.0, 1.0], 
                             distribution=["uniform", "gaussian"], sigma=[0.0, 1.0]), 
                                "all"),
        ("ga_crossover", dict(pairing=["evenodd", "rank", "cost", 
                                       "tournament_2_100", "tournament_2_75", 
                                       "tournament_3_100", "tournament_3_75"],
                             crossover=["single", "two", "uniform", "blend", 
                                        "linear_0.5_0.5", "linear_1.5_0.5", 
                                        "linear_0.5_1.5", "linear_1.5_1.5", 
                                        "linear_-0.5_0.5", "linear_0.5_-0.5"], 
                              mating_pool_factor=[0.0, 1.0]), "all")
        ]