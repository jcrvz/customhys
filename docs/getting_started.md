# Getting Started

This guide walks you through your first steps with CUSTOMHyS — from solving a simple
optimisation problem with a metaheuristic to running a full hyper-heuristic search.

## Prerequisites

- Python 3.10 or newer
- Basic familiarity with continuous optimisation concepts

## A Minimal Example

The simplest way to use CUSTOMHyS is to create a **Metaheuristic** that solves a
benchmark function:

```python
from customhys import benchmark_func as bf
from customhys.metaheuristic import Metaheuristic

# 1. Choose a benchmark problem (Sphere function in 5 dimensions)
problem = bf.Sphere(5)
prob = {
    "function": problem.get_func_val,
    "is_constrained": True,
    "boundaries": problem.get_search_range(),
}

# 2. Define a sequence of search operators
#    Each operator is a tuple: (name, parameters, selector)
search_operators = [
    ("random_search", {"scale": 0.01, "distribution": "uniform"}, "greedy"),
    ("swarm_dynamic", {
        "self_conf": 2.54,
        "swarm_conf": 2.56,
        "version": "inertial",
        "inertial_weight": 0.7,
    }, "all"),
]

# 3. Create and run the metaheuristic
mh = Metaheuristic(prob, search_operators, num_agents=30, num_iterations=100)
mh.run()

# 4. Retrieve the best solution found
position, fitness = mh.get_solution()
print(f"Best position: {position}")
print(f"Best fitness:  {fitness:.6e}")
```

## Running a Hyper-Heuristic

Instead of manually choosing operators, let the **Hyperheuristic** class discover the
best combination automatically:

```python
from customhys import benchmark_func as bf
from customhys.hyperheuristic import Hyperheuristic

# Define the problem
problem = bf.Rastrigin(10)
prob = {
    "function": problem.get_func_val,
    "is_constrained": True,
    "boundaries": problem.get_search_range(),
}

# Configure the hyper-heuristic
parameters = {
    "cardinality": 3,
    "num_iterations": 100,
    "num_agents": 30,
    "num_replicas": 30,
    "num_steps": 100,
    "stagnation_percentage": 0.3,
    "max_temperature": 200,
    "cooling_rate": 0.05,
}

# Run the hyper-heuristic (uses the default operator collection)
hh = Hyperheuristic(
    heuristic_space="default.txt",
    problem=prob,
    parameters=parameters,
)
hh.run()
```

The hyper-heuristic will evaluate many candidate metaheuristics and return the best
sequence of search operators for the given problem.

## Running a Full Experiment

For batch experiments across multiple benchmark functions and dimensions, use the
**Experiment** class with a JSON configuration file:

```python
from customhys.experiment import Experiment

exp = Experiment(config_file="demo.json")
exp.run()
```

See the {doc}`user_guide/index` for a deeper dive into each module.

## What's Next?

- {doc}`installation` — detailed installation options and extras
- {doc}`user_guide/architecture` — understand the module hierarchy
- {doc}`user_guide/tutorials` — hands-on tutorials (English & Spanish)
- {doc}`api/index` — full API reference
