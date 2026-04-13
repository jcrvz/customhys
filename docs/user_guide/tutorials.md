# Tutorials

This page provides narrative walkthroughs of the main CUSTOMHyS workflows.
Full interactive notebooks are available in the
[examples/](https://github.com/jcrvz/customhys/tree/master/examples) directory on
GitHub.

| Notebook | Language | Binder / GitHub |
|----------|----------|-----------------|
| `Tutorial_English.ipynb` | English | [Open on GitHub](https://github.com/jcrvz/customhys/blob/master/examples/Tutorial_English.ipynb) |
| `Tutorial_Spanish.ipynb` | Spanish | [Open on GitHub](https://github.com/jcrvz/customhys/blob/master/examples/Tutorial_Spanish.ipynb) |
| `quickTest.ipynb` | English | [Open on GitHub](https://github.com/jcrvz/customhys/blob/master/examples/quickTest.ipynb) |

---

## Tutorial 1 — Solving a Benchmark Function

### Step 1: Choose a problem

CUSTOMHyS ships with dozens of N-dimensional benchmark functions. Each function is a
class that exposes a uniform interface:

```python
from customhys import benchmark_func as bf

# List all available functions
print(bf.__all__[:10])
# ['Ackley1', 'Ackley4', 'Alpine1', 'Alpine2', 'Bohachevsky', ...]

# Instantiate a 10-dimensional Sphere function
func = bf.Sphere(10)

# Get the problem dictionary expected by Metaheuristic / Hyperheuristic
prob = {
    "function": func.get_func_val,
    "is_constrained": True,
    "boundaries": func.get_search_range(),
}
```

### Step 2: Pick search operators

Operators are specified as 3-tuples `(name, parameters_dict, selector_name)`. You can
inspect all available operators in {mod}`customhys.operators`.

```python
search_operators = [
    ("random_search", {"scale": 0.01, "distribution": "uniform"}, "greedy"),
    ("swarm_dynamic", {
        "self_conf": 2.54,
        "swarm_conf": 2.56,
        "version": "inertial",
        "inertial_weight": 0.7,
    }, "all"),
]
```

### Step 3: Create and run a metaheuristic

```python
from customhys.metaheuristic import Metaheuristic

mh = Metaheuristic(
    prob,
    search_operators,
    num_agents=30,
    num_iterations=100,
    verbose=True,
)
mh.run()

position, fitness = mh.get_solution()
print(f"Best fitness found: {fitness:.6e}")
```

---

## Tutorial 2 — Automatic Metaheuristic Design with the Hyper-heuristic

Instead of hand-picking operators, let the framework search the heuristic space for
you.

### Step 1: Prepare the problem (same as above)

```python
from customhys import benchmark_func as bf

func = bf.Rastrigin(10)
prob = {
    "function": func.get_func_val,
    "is_constrained": True,
    "boundaries": func.get_search_range(),
}
```

### Step 2: Configure the hyper-heuristic

```python
from customhys.hyperheuristic import Hyperheuristic

parameters = {
    "cardinality": 3,            # max operators per metaheuristic
    "num_iterations": 100,       # iterations each candidate MH runs
    "num_agents": 30,            # population size
    "num_replicas": 30,          # evaluations per candidate
    "num_steps": 100,            # hyper-heuristic search steps
    "stagnation_percentage": 0.3,
    "max_temperature": 200,
    "cooling_rate": 0.05,
}

hh = Hyperheuristic(
    heuristic_space="default.txt",
    problem=prob,
    parameters=parameters,
)
```

### Step 3: Run

```python
hh.run()
```

The hyper-heuristic evaluates different operator sequences using a Simulated Annealing
acceptance criterion and returns the best-performing metaheuristic.

---

## Tutorial 3 — Batch Experiments

Use the **Experiment** class to run many hyper-heuristic searches across problems and
dimensions.

### Using a JSON configuration file

Configuration files live in `customhys/exconf/`. Here is a minimal example:

```json
{
  "experiment_name": "my_experiment",
  "experiment_type": "default",
  "heuristic_collection_file": "default.txt",
  "weights_dataset_file": "operators_weights.json",
  "hh_config": {
    "cardinality": 3,
    "num_replicas": 30
  },
  "prob_config": {
    "dimensions": [2, 5, 10],
    "functions": ["Sphere", "Rastrigin", "Ackley1"]
  }
}
```

```python
from customhys.experiment import Experiment

exp = Experiment(config_file="my_experiment.json")
exp.run()
```

### Programmatic configuration

```python
exp = Experiment(
    exp_config={
        "experiment_name": "quick_test",
        "experiment_type": "default",
        "heuristic_collection_file": "default.txt",
    },
    hh_config={"cardinality": 3, "num_replicas": 30},
    prob_config={"dimensions": [2, 5], "functions": ["Sphere"]},
)
exp.run()
```

---

## Tutorial 4 — Creating Custom Operator Collections

Operator collections are plain-text files in `customhys/collections/`, with one
operator tuple per line:

```text
('random_search', {'scale': 0.01, 'distribution': 'uniform'}, 'greedy')
('swarm_dynamic', {'self_conf': 2.54, 'swarm_conf': 2.56, 'version': 'inertial', 'inertial_weight': 0.7}, 'all')
('differential_mutation', {'expression': 'rand', 'num_rands': 1, 'factor': 0.7}, 'greedy')
```

You can also generate collections programmatically:

```python
from customhys import operators as op

# Build all operator variants with 5 parameter discretisations
operators = op.obtain_operators(num_vals=5)
op.build_operators(operators, file_name="my_collection")
```

This writes `my_collection.txt` into `collections/`.
