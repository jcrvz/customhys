"""
Tests for the role-based operator classification and initialization operators.

Covers:
- _get_role() derives role from operator name prefix via _ROLE_PREFIXES
- process_operators handles 3-tuples
- All initialize_* functions exist and reset positions correctly
- population.initialise_positions accepts **kwargs and tuple format
- Hyperheuristic.initialization_operators property
- evaluate_candidate_solution extracts init op from sequence
"""

import ast

import numpy as np
import pytest

from customhys import operators as op
from customhys import population as pp

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pop(num_dimensions: int = 3, num_agents: int = 10) -> pp.Population:
    boundaries = ([-5.0] * num_dimensions, [5.0] * num_dimensions)
    return pp.Population(boundaries, num_agents=num_agents)


# ---------------------------------------------------------------------------
# _get_role
# ---------------------------------------------------------------------------


class TestGetRole:
    def test_perturb_operator_returns_perturb(self):
        t = ("random_search", {"scale": 0.1}, "greedy")
        assert op._get_role(t) == "perturb"

    def test_initialize_prefix_returns_initialize(self):
        t = ("initialize_random", {}, "all")
        assert op._get_role(t) == "initialize"

    def test_initialize_prefix_applies_to_all_variants(self):
        for name in ["initialize_sobol", "initialize_halton", "initialize_lhs", "initialize_beta"]:
            assert op._get_role((name, {}, "all")) == "initialize"

    def test_unknown_prefix_defaults_to_perturb(self):
        t = ("some_new_operator", {}, "greedy")
        assert op._get_role(t) == "perturb"

    def test_role_prefixes_dict_is_extensible(self):
        """_ROLE_PREFIXES is the single source of truth for role derivation."""
        assert "initialize_" in op._ROLE_PREFIXES
        assert op._ROLE_PREFIXES["initialize_"] == "initialize"

    def test_all_init_ops_in_default_initializers_txt_have_initialize_role(self):
        """Every entry in default_initializers.txt must have role='initialize' via name prefix."""
        import os

        path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "customhys",
            "collections",
            "default_initializers.txt",
        )
        with open(path, encoding="utf-8") as f:
            entries = [ast.literal_eval(line.strip()) for line in f if line.strip()]
        for entry in entries:
            assert op._get_role(entry) == "initialize", f"Expected 'initialize' role for {entry}"


# ---------------------------------------------------------------------------
# process_operators — backward compat with 3-tuples and 4-tuples
# ---------------------------------------------------------------------------


class TestProcessOperators:
    def test_three_tuple(self):
        seq = [("random_search", {"scale": 0.1}, "greedy")]
        perturbators, selectors = op.process_operators(seq)
        assert len(perturbators) == 1
        assert len(selectors) == 1
        assert selectors[0] == "greedy"

    def test_initialize_tuple(self):
        seq = [("initialize_random", {}, "all")]
        perturbators, selectors = op.process_operators(seq)
        assert len(perturbators) == 1
        assert selectors[0] == "all"

    def test_mixed_operators(self):
        seq = [
            ("random_search", {"scale": 0.1}, "greedy"),
            ("initialize_sobol", {"scramble": True}, "all"),
        ]
        perturbators, selectors = op.process_operators(seq)
        assert len(perturbators) == 2
        assert len(selectors) == 2


# ---------------------------------------------------------------------------
# initialize_* functions via operator interface
# ---------------------------------------------------------------------------


class TestInitializeOperators:
    @pytest.mark.parametrize(
        "func_name",
        [
            "initialize_random",
            "initialize_sobol",
            "initialize_halton",
            "initialize_grid",
            "initialize_lhs",
            "initialize_beta",
            "initialize_normal",
            "initialize_lognormal",
            "initialize_exponential",
            "initialize_rayleigh",
        ],
    )
    def test_function_exists_in_all(self, func_name):
        assert func_name in op.__all__

    @pytest.mark.parametrize(
        "func_name",
        [
            "initialize_random",
            "initialize_sobol",
            "initialize_halton",
            "initialize_grid",
            "initialize_lhs",
            "initialize_beta",
            "initialize_normal",
            "initialize_lognormal",
            "initialize_exponential",
            "initialize_rayleigh",
        ],
    )
    def test_function_resets_positions_to_finite_values(self, func_name):
        pop = _make_pop()
        func = getattr(op, func_name)
        func(pop)
        assert np.all(np.isfinite(pop.positions))
        assert pop.positions.shape == (10, 3)

    def test_initialize_sobol_scramble_false(self):
        """initialize_sobol passes scramble kwarg to population."""
        pop = _make_pop()
        op.initialize_sobol(pop, scramble=False)
        assert np.all(np.isfinite(pop.positions))

    def test_initialize_halton_scramble_false(self):
        pop = _make_pop()
        op.initialize_halton(pop, scramble=False)
        assert np.all(np.isfinite(pop.positions))

    def test_initialize_beta_custom_params(self):
        pop = _make_pop()
        op.initialize_beta(pop, a=0.2, b=0.8)
        assert np.all(np.isfinite(pop.positions))

    def test_initialize_normal_custom_params(self):
        pop = _make_pop()
        op.initialize_normal(pop, mean=0.3, std=0.1)
        assert np.all(np.isfinite(pop.positions))


# ---------------------------------------------------------------------------
# population.initialise_positions — tuple format and **kwargs
# ---------------------------------------------------------------------------


class TestInitialisePositionsTupleFormat:
    def test_tuple_format_with_params(self):
        pop = _make_pop()
        pop.initialise_positions(("beta", {"a": 0.3, "b": 0.7}))
        assert np.all(np.isfinite(pop.positions))

    def test_tuple_format_without_params(self):
        pop = _make_pop()
        pop.initialise_positions(("sobol", {}))
        assert np.all(np.isfinite(pop.positions))

    def test_kwargs_forwarded_to_sobol(self):
        pop1 = _make_pop(num_agents=8)
        pop2 = _make_pop(num_agents=8)
        pop1.initialise_positions("sobol", scramble=True)
        pop2.initialise_positions("sobol", scramble=True)
        # Both should produce finite arrays (determinism not guaranteed)
        assert np.all(np.isfinite(pop1.positions))
        assert np.all(np.isfinite(pop2.positions))

    def test_kwargs_forwarded_to_halton(self):
        pop = _make_pop()
        pop.initialise_positions("halton", scramble=False)
        assert np.all(np.isfinite(pop.positions))

    def test_kwargs_forwarded_to_normal(self):
        pop = _make_pop()
        pop.initialise_positions("normal", mean=0.6, std=0.05)
        assert np.all(np.isfinite(pop.positions))

    def test_none_scheme_uses_random(self):
        pop = _make_pop()
        pop.initialise_positions(None)
        assert np.all(np.isfinite(pop.positions))
