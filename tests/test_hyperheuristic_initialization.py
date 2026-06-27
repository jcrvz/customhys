"""
Tests for role-based operator classification in Hyperheuristic.

Covers:
- initialization_operators property
- ChangeInitOperator action in _choose_action and _obtain_candidate_solution
- evaluate_candidate_solution with init op at head (including bug regression)
- build_operators output (no role suffix — role is derived from name prefix)
- SA loop integration: HH selects and uses init ops correctly with real benchmark problems
"""

import ast

import numpy as np
import pytest

from customhys import benchmark_func as bf
from customhys import hyperheuristic as hh
from customhys import operators as op
from customhys.hyperheuristic import HyperheuristicError

# ---------------------------------------------------------------------------
# Shared operator tuples — 3-tuples only (role derived from name prefix)
# ---------------------------------------------------------------------------

INIT_OP_SOBOL = ("initialize_sobol", {"scramble": True}, "all")
INIT_OP_HALTON = ("initialize_halton", {"scramble": True}, "all")
INIT_OP_GRID = ("initialize_grid", {}, "all")
PERTURB_OP_RS = ("random_search", {"scale": 1.0, "distribution": "uniform"}, "greedy")
PERTURB_OP_LRW = (
    "local_random_walk",
    {"probability": 0.75, "scale": 1.0, "distribution": "gaussian"},
    "greedy",
)

MIXED_SPACE = [INIT_OP_SOBOL, INIT_OP_HALTON, PERTURB_OP_RS, PERTURB_OP_LRW]
# idx 0 = INIT_OP_SOBOL, idx 1 = INIT_OP_HALTON, idx 2 = PERTURB_OP_RS, idx 3 = PERTURB_OP_LRW

PURE_PERTURB_SPACE = [PERTURB_OP_RS, PERTURB_OP_LRW]

SINGLE_INIT_SPACE = [INIT_OP_SOBOL, PERTURB_OP_RS, PERTURB_OP_LRW]
# idx 0 = INIT_OP_SOBOL, idx 1 = PERTURB_OP_RS, idx 2 = PERTURB_OP_LRW

GRID_MIXED_SPACE = [INIT_OP_GRID, INIT_OP_SOBOL, PERTURB_OP_RS, PERTURB_OP_LRW]
# idx 0 = INIT_OP_GRID, idx 1 = INIT_OP_SOBOL, idx 2 = PERTURB_OP_RS, idx 3 = PERTURB_OP_LRW

# ---------------------------------------------------------------------------
# Minimal HH parameters — fast but functional
# ---------------------------------------------------------------------------

MINI_PARAMS = {
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


def make_hh(space, problem=None):
    """Create a minimal Hyperheuristic instance with the given heuristic space."""
    if problem is None:
        problem = bf.Sphere(variable_num=2).get_formatted_problem()
    return hh.Hyperheuristic(
        heuristic_space=space,
        problem=problem,
        parameters=MINI_PARAMS.copy(),
    )


# ===========================================================================
# 1. initialization_operators property
# ===========================================================================


class TestInitializationOperatorsProperty:
    def test_empty_when_no_init_ops(self):
        hyper = make_hh(PURE_PERTURB_SPACE)
        assert hyper.initialization_operators == []

    def test_returns_only_init_ops_from_mixed_space(self):
        hyper = make_hh(MIXED_SPACE)
        assert len(hyper.initialization_operators) == 2

    def test_returns_correct_indices(self):
        hyper = make_hh(MIXED_SPACE)
        for idx, op_tuple in hyper.initialization_operators:
            assert hyper.heuristic_space[idx] is op_tuple

    def test_single_init_op_returns_one_entry(self):
        hyper = make_hh(SINGLE_INIT_SPACE)
        assert len(hyper.initialization_operators) == 1

    def test_all_results_have_initialize_role(self):
        hyper = make_hh(MIXED_SPACE)
        for _, op_tuple in hyper.initialization_operators:
            assert op._get_role(op_tuple) == "initialize"


# ===========================================================================
# 2. _choose_action — ChangeInitOperator availability
# ===========================================================================


class TestChooseActionWithInitOps:
    def test_change_init_op_included_when_2_init_ops(self):
        hyper = make_hh(MIXED_SPACE)
        seen = {hyper._choose_action(2) for _ in range(200)}
        assert "ChangeInitOperator" in seen

    def test_change_init_op_excluded_when_1_init_op(self):
        hyper = make_hh(SINGLE_INIT_SPACE)
        for _ in range(50):
            assert hyper._choose_action(2) != "ChangeInitOperator"

    def test_change_init_op_excluded_when_no_init_ops(self):
        hyper = make_hh(PURE_PERTURB_SPACE)
        for _ in range(50):
            assert hyper._choose_action(2) != "ChangeInitOperator"


# ===========================================================================
# 3. _obtain_candidate_solution — ChangeInitOperator action
# ===========================================================================


class TestObtainCandidateSolutionChangeInitOperator:
    def test_preserves_sequence_length(self):
        hyper = make_hh(MIXED_SPACE)
        sol = np.array([0, 2, 3])
        result = hyper._obtain_candidate_solution(sol=sol, action="ChangeInitOperator")
        assert len(result) == len(sol)

    def test_tail_of_sequence_unchanged(self):
        hyper = make_hh(MIXED_SPACE)
        sol = np.array([0, 2, 3])
        result = hyper._obtain_candidate_solution(sol=sol, action="ChangeInitOperator")
        np.testing.assert_array_equal(result[1:], sol[1:])

    def test_first_element_is_a_valid_init_op_index(self):
        hyper = make_hh(MIXED_SPACE)
        init_indices = {i for i, _ in hyper.initialization_operators}
        sol = np.array([0, 2])
        result = hyper._obtain_candidate_solution(sol=sol, action="ChangeInitOperator")
        assert int(result[0]) in init_indices

    def test_new_first_element_can_differ_from_original(self):
        hyper = make_hh(MIXED_SPACE)
        sol = np.array([0, 2])
        seen_first = {int(hyper._obtain_candidate_solution(sol=sol, action="ChangeInitOperator")[0]) for _ in range(50)}
        assert 1 in seen_first


class TestHeadInitializerConstraint:
    def test_shift_keeps_initializer_at_head(self):
        hyper = make_hh(MIXED_SPACE)
        init_indices = {i for i, _ in hyper.initialization_operators}
        sol = np.array([0, 2, 3])
        result = hyper._obtain_candidate_solution(sol=sol, action="Shift")
        assert int(result[0]) in init_indices

    def test_swap_keeps_initializer_at_head(self):
        hyper = make_hh(MIXED_SPACE)
        init_indices = {i for i, _ in hyper.initialization_operators}
        sol = np.array([0, 2, 3])
        result = hyper._obtain_candidate_solution(sol=sol, action="Swap")
        assert int(result[0]) in init_indices

    def test_mirror_keeps_initializer_at_head(self):
        hyper = make_hh(MIXED_SPACE)
        init_indices = {i for i, _ in hyper.initialization_operators}
        sol = np.array([0, 2, 3])
        result = hyper._obtain_candidate_solution(sol=sol, action="Mirror")
        assert int(result[0]) in init_indices

    def test_roll_keeps_initializer_at_head(self):
        hyper = make_hh(MIXED_SPACE)
        init_indices = {i for i, _ in hyper.initialization_operators}
        sol = np.array([0, 2, 3])
        result = hyper._obtain_candidate_solution(sol=sol, action="Roll")
        assert int(result[0]) in init_indices

    def test_initial_solution_starts_with_initializer(self):
        hyper = make_hh(MIXED_SPACE)
        init_indices = {i for i, _ in hyper.initialization_operators}
        result = hyper._obtain_candidate_solution()
        assert int(result[0]) in init_indices


# ===========================================================================
# 4. evaluate_candidate_solution — init op extraction
# ===========================================================================


class TestEvaluateCandidateSolutionWithInitOp:
    def test_raises_when_sequence_is_only_init_op(self):
        hyper = make_hh(MIXED_SPACE)
        with pytest.raises(HyperheuristicError):
            hyper.evaluate_candidate_solution(np.array([0]))

    def test_valid_sequence_with_init_op_returns_performance(self):
        hyper = make_hh(MIXED_SPACE)
        perf, details = hyper.evaluate_candidate_solution(np.array([0, 2]))
        assert isinstance(perf, float)
        assert "fitness" in details

    def test_pure_perturb_sequence_still_works(self):
        hyper = make_hh(MIXED_SPACE)
        perf, details = hyper.evaluate_candidate_solution(np.array([2, 3]))
        assert isinstance(perf, float)
        assert "fitness" in details

    def test_sobol_init_op_produces_finite_performance(self):
        hyper = make_hh(MIXED_SPACE)
        perf, details = hyper.evaluate_candidate_solution(np.array([0, 2]))
        assert np.isfinite(perf)
        assert all(np.isfinite(f) for f in details["fitness"])

    def test_grid_init_op_does_not_raise(self):
        hyper = make_hh(GRID_MIXED_SPACE)
        perf, _ = hyper.evaluate_candidate_solution(np.array([0, 2]))
        assert isinstance(perf, float)

    def test_initializer_in_tail_is_allowed(self):
        hyper = make_hh(MIXED_SPACE)
        perf, details = hyper.evaluate_candidate_solution(np.array([0, 1, 2]))
        assert isinstance(perf, float)
        assert all(np.isfinite(f) for f in details["fitness"])


# ===========================================================================
# 5. build_operators — no role suffix in generated file (role from name prefix)
# ===========================================================================


class TestBuildOperatorsOutput:
    def test_init_operator_writes_no_role_suffix(self, tmp_path, monkeypatch):
        """build_operators writes clean 3-tuples; role is derived from name, not stored."""
        monkeypatch.chdir(tmp_path)
        heuristics = [("initialize_random", {}, ["all"])]
        op.build_operators(heuristics=heuristics, file_name="test_init")
        content = (tmp_path / "collections" / "test_init.txt").read_text(encoding="utf-8")
        assert "'initialize'" not in content
        assert "'perturb'" not in content

    def test_perturb_operator_writes_no_role_suffix(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        heuristics = [("random_search", {"scale": [1.0]}, ["greedy"])]
        op.build_operators(heuristics=heuristics, file_name="test_perturb")
        content = (tmp_path / "collections" / "test_perturb.txt").read_text(encoding="utf-8")
        assert "'initialize'" not in content
        assert "'perturb'" not in content

    def test_generated_file_entries_have_correct_role_via_prefix(self, tmp_path, monkeypatch):
        """Entries written by build_operators must have the correct role when read back."""
        monkeypatch.chdir(tmp_path)
        heuristics = [
            ("initialize_lhs", {}, ["all"]),
            ("random_search", {"scale": [1.0]}, ["greedy"]),
        ]
        op.build_operators(heuristics=heuristics, file_name="test_mixed")
        content = (tmp_path / "collections" / "test_mixed.txt").read_text(encoding="utf-8")
        entries = [ast.literal_eval(line.strip()) for line in content.splitlines() if line.strip()]
        roles = [op._get_role(e) for e in entries]
        assert roles[0] == "initialize"
        assert roles[1] == "perturb"


# ===========================================================================
# 6. SA loop integration — HH selects and uses init ops with real problems
# ===========================================================================


@pytest.mark.parametrize(
    "problem_cls,dims",
    [
        (bf.Sphere, 2),
        (bf.Rastrigin, 2),
        (bf.Ackley1, 2),
    ],
)
class TestHyperheuristicSolveWithInitOps:
    """
    Integration tests: run the full SA loop and verify that initializer operators
    are selected as part of the best solution.
    """

    def test_solve_completes_without_error(self, problem_cls, dims):
        problem = problem_cls(variable_num=dims).get_formatted_problem()
        hyper = make_hh(MIXED_SPACE, problem=problem)
        encoded, performance, _, _ = hyper.solve(save_steps=False)
        assert encoded is not None
        assert isinstance(performance, float)

    def test_solve_returns_finite_performance(self, problem_cls, dims):
        problem = problem_cls(variable_num=dims).get_formatted_problem()
        hyper = make_hh(MIXED_SPACE, problem=problem)
        _, performance, _, _ = hyper.solve(save_steps=False)
        assert np.isfinite(performance)

    def test_best_encoded_solution_starts_with_init_op(self, problem_cls, dims):
        """The encoded solution from the SA loop must have an init op at position 0."""
        problem = problem_cls(variable_num=dims).get_formatted_problem()
        hyper = make_hh(MIXED_SPACE, problem=problem)
        init_indices = {i for i, _ in hyper.initialization_operators}
        encoded, _, _, _ = hyper.solve(save_steps=False)
        assert (
            int(encoded[0]) in init_indices
        ), f"Expected encoded[0] to be an init op index {init_indices}, got {encoded[0]}"

    def test_best_solution_operator_names_include_init_prefix(self, problem_cls, dims):
        """The decoded best solution must contain an initialize_* operator at position 0."""
        problem = problem_cls(variable_num=dims).get_formatted_problem()
        hyper = make_hh(MIXED_SPACE, problem=problem)
        encoded, _, _, _ = hyper.solve(save_steps=False)
        first_op_name = hyper.get_operators(encoded.tolist())[0][0]
        assert first_op_name.startswith(
            "initialize_"
        ), f"Expected first operator to start with 'initialize_', got '{first_op_name}'"


class TestHyperheuristicSolveWithoutInitOps:
    """Ensure backward compatibility: pure perturb space still works normally."""

    def test_solve_without_init_ops_completes(self):
        problem = bf.Sphere(variable_num=2).get_formatted_problem()
        hyper = make_hh(PURE_PERTURB_SPACE, problem=problem)
        _, performance, _, _ = hyper.solve(save_steps=False)
        assert np.isfinite(performance)

    def test_encoded_solution_has_no_init_op_when_space_is_pure_perturb(self):
        problem = bf.Sphere(variable_num=2).get_formatted_problem()
        hyper = make_hh(PURE_PERTURB_SPACE, problem=problem)
        encoded, _, _, _ = hyper.solve(save_steps=False)
        init_indices = {i for i, _ in hyper.initialization_operators}
        assert len(init_indices) == 0
        for idx in encoded:
            assert op._get_role(hyper.heuristic_space[int(idx)]) == "perturb"
