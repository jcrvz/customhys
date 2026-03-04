import numpy as np
import pytest

from customhys import population as pp


def _boundaries(num_dimensions: int = 3) -> tuple[list[float], list[float]]:
    return ([-100.0] * num_dimensions, [100.0] * num_dimensions)


class TestPopulationInitializationSchemes:
    @pytest.mark.parametrize(
        "scheme",
        ["lhs", "sobol", "halton", "beta", "normal", "lognormal", "exponential", "rayleigh"],
    )
    def test_new_schemes_shape_bounds_and_finite(self, scheme):
        """All new initialisation schemes must return finite values in [-1, 1] with expected shape."""
        num_agents = 64
        num_dimensions = 4
        pop = pp.Population(_boundaries(num_dimensions), num_agents=num_agents)

        pop.initialise_positions(scheme)
        positions = pop.positions

        assert positions.shape == (num_agents, num_dimensions)
        assert np.all(np.isfinite(positions))
        assert np.all(positions >= -1.0)
        assert np.all(positions <= 1.0)

    def test_lhs_covers_all_strata_in_each_dimension(self):
        """LHS should place one sample in each interval per dimension."""
        num_agents = 32
        num_dimensions = 3
        pop = pp.Population(_boundaries(num_dimensions), num_agents=num_agents)

        pop.initialise_positions("lhs")
        sample_01 = (pop.positions + 1.0) / 2.0

        for dimension in range(num_dimensions):
            bins = np.floor(sample_01[:, dimension] * num_agents).astype(int)
            bins = np.clip(bins, 0, num_agents - 1)
            assert np.array_equal(np.sort(bins), np.arange(num_agents))

    def test_sobol_keeps_requested_number_of_agents_for_non_power_of_two(self):
        """Sobol generator returns exactly num_agents rows even when num_agents is not a power of two."""
        num_agents = 30
        pop = pp.Population(_boundaries(3), num_agents=num_agents)

        pop.initialise_positions("sobol")

        assert pop.positions.shape == (num_agents, 3)
        assert np.unique(pop.positions, axis=0).shape[0] == num_agents

    def test_beta_has_more_edge_samples_than_uniform_random(self):
        """Beta(0.5, 0.5) should concentrate more points near boundaries than uniform random."""
        num_agents = 4000
        num_dimensions = 1

        np.random.seed(123)
        uniform_pop = pp.Population(_boundaries(num_dimensions), num_agents=num_agents)
        uniform_pop.initialise_positions("random")
        uniform_edge_ratio = np.mean(np.abs(uniform_pop.positions) > 0.8)

        np.random.seed(123)
        beta_pop = pp.Population(_boundaries(num_dimensions), num_agents=num_agents)
        beta_pop.initialise_positions("beta")
        beta_edge_ratio = np.mean(np.abs(beta_pop.positions) > 0.8)

        assert beta_edge_ratio > uniform_edge_ratio

    @pytest.mark.parametrize(
        ("scheme", "expectation"),
        [
            ("normal", lambda mean: abs(mean) < 0.12),
            ("exponential", lambda mean: mean < 0.0),
            ("lognormal", lambda mean: mean > 0.0),
        ],
    )
    def test_distributional_bias_for_selected_schemes(self, scheme, expectation):
        """Selected distribution-based schemes should show expected mean bias after mapping to [-1, 1]."""
        np.random.seed(7)
        pop = pp.Population(_boundaries(1), num_agents=5000)

        pop.initialise_positions(scheme)
        mean_value = float(np.mean(pop.positions))

        assert expectation(mean_value)

    def test_scheme_name_is_normalized(self):
        """Scheme names should be case-insensitive and whitespace-tolerant."""
        pop = pp.Population(_boundaries(2), num_agents=20)
        pop.initialise_positions("  LHS  ")
        assert pop.positions.shape == (20, 2)

    def test_unknown_scheme_falls_back_to_random(self):
        """Unknown scheme should fallback to random initialisation without failing."""
        pop = pp.Population(_boundaries(2), num_agents=25)
        pop.initialise_positions("nonexistent_scheme")
        assert pop.positions.shape == (25, 2)
        assert np.all(pop.positions >= -1.0)
        assert np.all(pop.positions <= 1.0)
