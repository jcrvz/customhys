# save as: scripts/smoke_hh_solvers.py
from copy import deepcopy

import numpy as np

from customhys import benchmark_func as bf
from customhys import hyperheuristic as hhmod
from customhys.experiment import read_config_file

# Default config completa (evita KeyError de params faltando)
_, base_params, _ = read_config_file()

base_params.update(
    {
        "cardinality": 2,
        "cardinality_min": 1,
        "num_agents": 8,
        "num_iterations": 5,
        "num_replicas": 3,
        "num_steps": 5,
        "as_mh": False,
        "verbose": False,
        "verbose_statistics": False,
        "trial_overflow": True,
        "initial_scheme": "random",
    }
)

problem = bf.Sphere(2).get_formatted_problem()
heuristics = [
    ("random_search", {"scale": 1.0, "distribution": "uniform"}, "greedy"),
    ("local_random_walk", {"probability": 0.75, "scale": 0.5, "distribution": "gaussian"}, "greedy"),
]


def run_mode(mode: str):
    params = deepcopy(base_params)
    params["solver"] = mode

    if mode == "neural_network":
        if not hhmod._using_tensorflow:
            print("neural_network: SKIPPED (tensorflow não instalado)")
            return
        params["tabu_idx"] = 2
        params["model_params"] = {
            "model_architecture": "MLP",
            "model_architecture_layers": [(8, "relu", "Dense")],
            "sample_params": {
                "retrieve_sequences": False,
                "limit_seqs": 8,
                "random": 0.5,
                "store_sequences": False,
            },
            "fitness_to_weight": "rank",
            "encoder": "identity",
            "memory_length": 5,
            "epochs": 1,
            "load_model": False,
            "save_model": False,
        }

    hh = hhmod.Hyperheuristic(
        heuristic_space=heuristics,
        problem=problem,
        parameters=params,
        file_label=f"smoke-{mode}",
    )

    out = hh.solve(mode, save_steps=False)

    if mode == "static":
        best_sol, best_perf, hist_curr, hist_best = out
        assert len(best_sol) > 0
        assert np.isfinite(best_perf)
        assert len(hist_curr) > 0 and len(hist_best) > 0
    elif mode == "dynamic":
        fitness_per_rep, sequence_per_rep, transition_matrix = out
        assert len(fitness_per_rep) == params["num_replicas"]
        assert len(sequence_per_rep) == params["num_replicas"]
        assert transition_matrix is None or len(transition_matrix) >= 0
    elif mode == "neural_network":
        fitness_per_rep, sequence_per_rep = out
        assert len(fitness_per_rep) == params["num_replicas"]
        assert len(sequence_per_rep) == params["num_replicas"]

    print(f"{mode}: OK")


for solver_mode in ["static", "dynamic", "neural_network"]:
    run_mode(solver_mode)
