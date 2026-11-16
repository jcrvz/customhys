"""
Regression tests: the shipped default collection must stay compatible with the shipped precomputed weights.

``operators_weights.json`` holds one weight per perturbation operator (205). ``default.txt`` also lists the
``initialize_*`` operators, which are kept after the perturbation operators so that the original indices are stable.
"""

import ast
from pathlib import Path

import numpy as np
import pytest

from customhys import benchmark_func as bf
from customhys import hyperheuristic as hh
from customhys import operators as op
from customhys import tools as tl
from customhys.hyperheuristic import HyperheuristicError

COLLECTIONS = Path(hh.__file__).parent / "collections"
_LINES = (COLLECTIONS / "default.txt").read_text(encoding="utf-8").splitlines()
SPACE = [ast.literal_eval(line) for line in _LINES if line.strip()]
NUM_PERTURB = sum(op._get_role(o) != "initialize" for o in SPACE)
PERTURB_SPACE = SPACE[:NUM_PERTURB]

PARAMS = {
    "cardinality": 3,
    "cardinality_min": 1,
    "num_replicas": 1,
    "num_iterations": 2,
    "num_agents": 3,
    "as_mh": False,
    "num_steps": 3,
    "stagnation_percentage": 0.37,
    "max_temperature": 1.0,
    "min_temperature": 1e-6,
    "cooling_rate": 1e-3,
    "temperature_scheme": "fast",
    "acceptance_scheme": "exponential",
    "allow_weight_matrix": True,
    "trial_overflow": False,
    "learnt_dataset": None,
    "repeat_operators": True,
    "verbose": False,
    "learning_portion": 0.37,
    "solver": "static",
}


def _weights_for(problem):
    data = tl.read_json(str(COLLECTIONS / "operators_weights.json"))
    return data["2"][problem.get_features(fmt="string", wrd="1")]


def _make(space, weights):
    problem = bf.Sphere(2)
    return hh.Hyperheuristic(
        heuristic_space=space,
        problem=problem.get_formatted_problem(),
        parameters=dict(PARAMS),
        weights_array=weights,
    )


def test_perturbation_operators_keep_their_original_indices():
    """Initializers must come after the perturbation operators (stable indices for stored weights/sequences)."""
    roles = [op._get_role(o) for o in SPACE]
    assert roles == ["perturb"] * NUM_PERTURB + ["initialize"] * (len(SPACE) - NUM_PERTURB)


def test_shipped_weights_work_with_default_collection():
    weights = _weights_for(bf.Sphere(2))
    assert len(weights) == NUM_PERTURB < len(SPACE)

    np.random.seed(0)
    h = _make(SPACE, weights)
    assert h.weights.shape == (len(SPACE),)
    assert np.isclose(h.weights.sum(), 1.0)
    # Weights remain attached to the same perturbation operators
    np.testing.assert_allclose(h.weights[:NUM_PERTURB], weights)
    assert np.all(h.weights[NUM_PERTURB:] == 0.0)

    candidate = h._obtain_candidate_solution(sol=3)
    assert len(candidate) == 3
    assert candidate[0] in [i for i, _ in h.initialization_operators]  # position 0 is an initializer
    assert all(op._get_role(SPACE[i]) != "initialize" for i in candidate[1:])  # weighted draws skip initializers


def test_weights_for_space_without_initializers_are_unchanged():
    weights = np.asarray(_weights_for(bf.Sphere(2)))
    h = _make(PERTURB_SPACE, weights)
    np.testing.assert_array_equal(h.weights, weights)


def test_weights_with_wrong_size_are_rejected():
    with pytest.raises(HyperheuristicError, match="weights_array"):
        _make(SPACE, np.ones(NUM_PERTURB - 1) / (NUM_PERTURB - 1))
