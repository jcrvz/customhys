# Architecture

CUSTOMHyS is organised into a hierarchy of modules that mirror the three levels of the
hyper-heuristic methodology: **low**, **mid**, and **high** level heuristics.

```{image} ../../docfiles/dependency_diagram.png
:alt: Module dependency diagram
:width: 100%
:align: center
```

---

## Module Overview

### 🤯 Benchmark Functions (`benchmark_func`)

A library of N-dimensional continuous benchmark functions used to evaluate optimisation
methods. Every function is implemented as a class with a unified interface
(`get_func_val`, `get_search_range`, `get_optimal`).

The catalogue includes classical families such as Ackley, Rastrigin, Rosenbrock,
Schwefel, Griewank, and many more — plus the CEC 2005 suite when the `optproblems`
package is available.

```python
from customhys import benchmark_func as bf

func = bf.Rastrigin(10)          # 10-dimensional Rastrigin
bounds = func.get_search_range() # ([-5.12, ...], [5.12, ...])
f_opt  = func.get_optimal()      # known global minimum value
```

**Source:** {mod}`customhys.benchmark_func`

---

### 👯‍♂️ Population (`population`)

A `Population` object represents a set of agents in a search space. It manages
positions, velocities, fitness values, and selection mechanisms. Populations do **not**
decide how to move — that is the job of search operators — but they know *when* to
accept a new position via a selector (greedy, metropolis, probabilistic, or all).

Key responsibilities:

- Initialise agents with different schemes (random, Sobol, Halton, LHS, …)
- Evaluate fitness for all agents
- Update global / particular / population best positions
- Constrain agents to the feasible domain

```python
from customhys.population import Population

pop = Population(boundaries=([-5]*10, [5]*10), num_agents=30)
pop.initialise_positions("random")
```

**Source:** {mod}`customhys.population`

---

### 🦾 Search Operators (`operators`)

A collection of 13 search operators (perturbators) extracted from well-known
metaheuristics. Each operator is a function that modifies the population's positions
*in place*. Operators are the building blocks that the hyper-heuristic combines to
assemble new metaheuristics.

Available operators:

| Operator | Alias | Origin |
|----------|-------|--------|
| `random_search` | RS | Random Search |
| `random_sample` | RX | Random Sampling |
| `random_flight` | RF | Lévy Flights |
| `local_random_walk` | RW | Random Walk |
| `central_force_dynamic` | CF | Central Force Optimisation |
| `differential_mutation` | DM | Differential Evolution |
| `firefly_dynamic` | FD | Firefly Algorithm |
| `genetic_crossover` | GC | Genetic Algorithm |
| `genetic_mutation` | GM | Genetic Algorithm |
| `gravitational_search` | GS | Gravitational Search |
| `spiral_dynamic` | SD | Spiral Dynamics |
| `swarm_dynamic` | PS | Particle Swarm Optimisation |
| `linear_system` | LS | Linear System |

Each operator is paired with a **selector** that decides how agents accept the
proposed move: `greedy`, `all`, `metropolis`, or `probabilistic`.

```python
("swarm_dynamic", {"self_conf": 2.54, "swarm_conf": 2.56, "version": "inertial"}, "greedy")
```

**Source:** {mod}`customhys.operators`

---

### 🤖 Metaheuristic (`metaheuristic`)

A `Metaheuristic` object takes a problem definition and a *sequence* of
`(operator, parameters, selector)` tuples and runs a population-based search.

```python
from customhys.metaheuristic import Metaheuristic

mh = Metaheuristic(prob, search_operators, num_agents=30, num_iterations=100)
mh.run()
position, fitness = mh.get_solution()
```

Internally, at each iteration the metaheuristic applies each operator in sequence,
evaluates fitness, updates the population using the corresponding selector, and tracks
historical data (best fitness, centroid, radius).

**Source:** {mod}`customhys.metaheuristic`

---

### 👽 Hyper-heuristic (`hyperheuristic`)

The `Hyperheuristic` class sits one level above metaheuristics. Given a *collection*
of search operators (the heuristic space), it explores which combination — and in what
order — produces the best metaheuristic for a specific problem.

The exploration is driven by a Simulated Annealing strategy that controls the
acceptance of candidate metaheuristics. Key parameters include the number of replicas,
the cooling schedule, and the stagnation criterion.

When TensorFlow is available, the hyper-heuristic can also leverage a neural-network
model to predict promising operator sequences (see {mod}`customhys.machine_learning`).

**Source:** {mod}`customhys.hyperheuristic`

---

### 🏭 Experiment (`experiment`)

The `Experiment` class orchestrates batch runs of hyper-heuristic procedures across
multiple benchmark functions and dimensionalities. Configuration is provided via JSON
files stored in the `exconf/` directory.

```python
from customhys.experiment import Experiment

exp = Experiment(config_file="demo.json")
exp.run()
```

Experiments can be parallelised across CPU cores automatically.

**Source:** {mod}`customhys.experiment`

---

### 🗜️ Tools (`tools`)

Utility functions shared across the framework: JSON I/O, statistical summaries,
meta-skeleton printing, operator weight processing, and more.

**Source:** {mod}`customhys.tools`

---

### 🧠 Machine Learning (`machine_learning`)

Wrappers for TensorFlow-based neural network models that learn to predict good
operator sequences from historical hyper-heuristic data. This module provides:

- `DatasetSequences` — processes raw operator sequences and fitness data into training
  samples.
- `ModelPredictor` — a configurable feed-forward neural network for sequence
  prediction.

**Source:** {mod}`customhys.machine_learning`

---

### 🌡️ Characterisation (`characterisation`) — *work in progress*

Metrics for characterising benchmark function landscapes (e.g., via Lévy-walk
sampling and kernel density estimation).

**Source:** {mod}`customhys.characterisation`

---

### 📊 Visualisation (`visualisation`) — *work in progress*

Plotting utilities for experiment results (violin plots of fitness distributions,
performance overviews).

**Source:** {mod}`customhys.visualisation`
