# Data Structure

Experiments are saved as JSON files. This page documents the schema of those files so
you can post-process results or feed them to other tools.

## Top-Level Schema

```text
data_frame = {dict: N}
├── 'problem'    = {list: N}   — benchmark function names (str)
├── 'dimensions' = {list: N}   — problem dimensionalities (int)
└── 'results'    = {list: N}   — one results dict per problem
```

where **N** is the number of problem–dimension pairs in the experiment.

## Results Entry

Each element inside `'results'` has the following structure:

```text
results[i] = {dict: 6}
├── 'iteration'        = {list: M}  — HH iteration indices (int)
├── 'time'             = {list: M}  — wall-clock time per iteration (float, seconds)
├── 'performance'      = {list: M}  — best fitness found at each step (float)
├── 'encoded_solution' = {list: M}  — encoded operator indices (int)
├── 'solution'         = {list: M}  — decoded operator sequences
│   └── [i] = {list: C}            — one metaheuristic (C operators)
│       └── [j] = {list: 3}        — a search-operator tuple
│           ├── operator_name      (str)
│           ├── control_parameters (dict: P)
│           └── selector           (str)
└── 'details'          = {list: M}  — per-replica execution details
    └── [i] = {dict: 4}
        ├── 'fitness'    = {list: R}  — final fitness per replica (float)
        ├── 'positions'  = {list: R}  — final best positions (list: D, float)
        ├── 'historical' = {list: R}  — per-replica history
        │   └── [j] = {dict: 5}
        │       ├── 'fitness'     = {list: I}  (float)
        │       ├── 'positions'   = {list: I}  (list: D, float)
        │       ├── 'centroid'    = {list: I}  (list: D, float)
        │       ├── 'radius'      = {list: I}  (float)
        │       └── 'stagnation'  = {list: I}  (int)
        └── 'statistics' = {dict: 10}
            ├── 'nob' — number of observations (int)
            ├── 'Min' — minimum fitness (float)
            ├── 'Max' — maximum fitness (float)
            ├── 'Avg' — mean fitness (float)
            ├── 'Std' — standard deviation (float)
            ├── 'Skw' — skewness (float)
            ├── 'Kur' — kurtosis (float)
            ├── 'IQR' — interquartile range (float)
            ├── 'Med' — median (float)
            └── 'MAD' — median absolute deviation (float)
```

## Symbol Legend

| Symbol | Meaning |
|--------|---------|
| **N** | Number of problem–dimension pairs |
| **M** | Number of hyper-heuristic iterations (candidate metaheuristics evaluated) |
| **C** | Cardinality — number of search operators per metaheuristic |
| **P** | Number of control parameters for each search operator |
| **R** | Number of replicas per candidate metaheuristic |
| **D** | Dimensionality of the problem |
| **I** | Number of iterations the metaheuristic performs |

## Example: Loading Results

```python
from customhys import tools as tl

data = tl.read_json("data_files/raw/my_experiment.json")

# Iterate over problems
for i, problem_name in enumerate(data["problem"]):
    dims = data["dimensions"][i]
    best_perf = data["results"][i]["performance"][-1]
    print(f"{problem_name} (D={dims}): best fitness = {best_perf:.6e}")
```
