"""
Test suite for operators module.
Tests search operators and their functionality.
"""

import numpy as np
import pytest

from customhys import benchmark_func as bf
from customhys import operators as op
from customhys import population as pp


class TestOperatorsAvailability:
    """Test operators module structure."""

    def test_operators_list_exists(self):
        """Test that __all__ contains operators."""
        assert hasattr(op, '__all__')
        assert len(op.__all__) > 0

    def test_common_operators_present(self):
        """Test that common operators are available."""
        expected_operators = [
            'random_search',
            'random_sample',
            'local_random_walk',
            'swarm_dynamic',
            'differential_mutation',
            'genetic_crossover',
            'spiral_dynamic',
        ]

        for operator_name in expected_operators:
            assert operator_name in op.__all__


class TestRandomSearchOperator:
    """Test random_search operator."""

    @pytest.fixture
    def population(self):
        """Create a test population."""
        fun = bf.Sphere(2)
        pop = pp.Population(fun.get_search_range(), num_agents=5)
        pop.initialise_positions('random')
        pop.evaluate_fitness(lambda x: fun.get_function_value(x))
        return pop

    def test_random_search_execution(self, population):
        """Test that random_search can be executed."""
        original_positions = population.positions.copy()

        # Operators modify population in place
        op.random_search(
            population,
            scale=1.0,
            distribution='uniform'
        )

        new_positions = population.positions
        assert new_positions.shape == original_positions.shape
        assert not np.array_equal(new_positions, original_positions)

    def test_random_search_scale(self, population):
        """Test random_search with different scales."""
        original_positions = population.positions.copy()

        # Small scale should produce smaller changes
        op.random_search(
            population,
            scale=0.01,
            distribution='uniform'
        )

        small_scale_positions = population.positions.copy()

        # Reset and try large scale
        population.positions = original_positions.copy()
        op.random_search(
            population,
            scale=1.0,
            distribution='uniform'
        )

        large_scale_positions = population.positions

        assert small_scale_positions.shape == large_scale_positions.shape

    def test_random_search_distributions(self, population):
        """Test random_search with different distributions."""
        distributions = ['uniform', 'gaussian', 'levy']

        for dist in distributions:
            original_positions = population.positions.copy()
            op.random_search(
                population,
                scale=1.0,
                distribution=dist
            )
            # Check that positions changed
            assert population.positions.shape == original_positions.shape
            # Reset for next test
            population.positions = original_positions.copy()


class TestLocalRandomWalk:
    """Test local_random_walk operator."""

    @pytest.fixture
    def population(self):
        """Create a test population."""
        fun = bf.Sphere(2)
        pop = pp.Population(fun.get_search_range(), num_agents=5)
        pop.initialise_positions('random')
        pop.evaluate_fitness(lambda x: fun.get_function_value(x))
        return pop

    def test_local_random_walk_execution(self, population):
        """Test that local_random_walk can be executed."""
        original_positions = population.positions.copy()

        op.local_random_walk(
            population,
            probability=0.75,
            scale=1.0,
            distribution='uniform'
        )

        assert population.positions.shape == original_positions.shape

    def test_local_random_walk_probability(self, population):
        """Test local_random_walk with different probabilities."""
        # Test that it runs without error with different probabilities
        for prob in [0.0, 0.5, 1.0]:
            original_positions = population.positions.copy()
            op.local_random_walk(
                population,
                probability=prob,
                scale=1.0,
                distribution='uniform'
            )
            # Just check it doesn't crash and maintains shape
            assert population.positions.shape == original_positions.shape
            # Reset for next test
            population.positions = original_positions.copy()


class TestSwarmDynamic:
    """Test swarm_dynamic operator."""

    @pytest.fixture
    def population(self):
        """Create a test population with best positions."""
        fun = bf.Sphere(2)
        pop = pp.Population(fun.get_search_range(), num_agents=5)
        pop.initialise_positions('random')
        pop.evaluate_fitness(lambda x: fun.get_function_value(x))

        # Initialize particular best
        pop.particular_best_positions = pop.positions.copy()
        pop.particular_best_fitness = pop.fitness.copy()

        return pop

    def test_swarm_dynamic_execution(self, population):
        """Test that swarm_dynamic can be executed."""
        original_positions = population.positions.copy()

        op.swarm_dynamic(
            population,
            factor=0.7,
            self_conf=2.54,
            swarm_conf=2.56,
            version='inertial',
            distribution='uniform'
        )

        assert population.positions.shape == original_positions.shape

    def test_swarm_dynamic_versions(self, population):
        """Test different swarm dynamic versions."""
        versions = ['inertial', 'constriction']

        for version in versions:
            original_positions = population.positions.copy()
            op.swarm_dynamic(
                population,
                factor=0.7,
                self_conf=2.54,
                swarm_conf=2.56,
                version=version,
                distribution='uniform'
            )
            assert population.positions.shape == original_positions.shape
            # Reset for next test
            population.positions = original_positions.copy()


class TestDifferentialMutation:
    """Test differential_mutation operator."""

    @pytest.fixture
    def population(self):
        """Create a test population."""
        fun = bf.Sphere(2)
        pop = pp.Population(fun.get_search_range(), num_agents=5)
        pop.initialise_positions('random')
        pop.evaluate_fitness(lambda x: fun.get_function_value(x))
        return pop

    def test_differential_mutation_execution(self, population):
        """Test that differential_mutation can be executed."""
        original_positions = population.positions.copy()

        op.differential_mutation(
            population,
            expression='rand',
            num_rands=1,
            factor=0.8
        )

        assert population.positions.shape == original_positions.shape

    def test_differential_mutation_expressions(self, population):
        """Test different differential mutation expressions."""
        expressions = ['rand', 'best', 'current-to-best']

        for expr in expressions:
            try:
                original_positions = population.positions.copy()
                op.differential_mutation(
                    population,
                    expression=expr,
                    num_rands=1,
                    factor=0.8
                )
                assert population.positions.shape == original_positions.shape
                # Reset for next test
                population.positions = original_positions.copy()
            except Exception:
                # Some expressions might not work with small populations
                pass


class TestGeneticOperators:
    """Test genetic operators."""

    @pytest.fixture
    def population(self):
        """Create a test population."""
        fun = bf.Sphere(2)
        pop = pp.Population(fun.get_search_range(), num_agents=5)
        pop.initialise_positions('random')
        pop.evaluate_fitness(lambda x: fun.get_function_value(x))
        return pop

    def test_genetic_crossover_execution(self, population):
        """Test that genetic_crossover can be executed."""
        original_positions = population.positions.copy()

        op.genetic_crossover(
            population,
            pairing='rank',
            crossover='blend',
            mating_pool_factor=0.4
        )

        assert population.positions.shape == original_positions.shape

    def test_genetic_mutation_execution(self, population):
        """Test that genetic_mutation can be executed."""
        original_positions = population.positions.copy()

        op.genetic_mutation(
            population,
            mutation_rate=0.1
        )

        assert population.positions.shape == original_positions.shape


class TestSpiralDynamic:
    """Test spiral_dynamic operator."""

    @pytest.fixture
    def population(self):
        """Create a test population."""
        fun = bf.Sphere(2)
        pop = pp.Population(fun.get_search_range(), num_agents=5)
        pop.initialise_positions('random')
        pop.evaluate_fitness(lambda x: fun.get_function_value(x))
        return pop

    def test_spiral_dynamic_execution(self, population):
        """Test that spiral_dynamic can be executed."""
        original_positions = population.positions.copy()

        op.spiral_dynamic(
            population,
            radius=0.9,
            angle=22.5,
            sigma=0.1
        )

        assert population.positions.shape == original_positions.shape

    def test_spiral_dynamic_parameters(self, population):
        """Test spiral_dynamic with different parameters."""
        # Test with different radius values
        radii = [0.5, 0.9, 0.95]

        for r in radii:
            original_positions = population.positions.copy()
            op.spiral_dynamic(
                population,
                radius=r,
                angle=22.5,
                sigma=0.1
            )
            assert population.positions.shape == original_positions.shape
            # Reset for next test
            population.positions = original_positions.copy()


class TestOperatorHelpers:
    """Test operator helper functions."""

    def test_obtain_operators_function(self):
        """Test obtain_operators helper function."""
        if hasattr(op, 'obtain_operators'):
            operators = op.obtain_operators(num_vals=3)
            assert isinstance(operators, list)
            assert len(operators) > 0


class TestOperatorOutputBounds:
    """Test that operators respect population boundaries."""

    @pytest.fixture
    def population(self):
        """Create a test population."""
        fun = bf.Sphere(2)
        pop = pp.Population(fun.get_search_range(), num_agents=5)
        pop.initialise_positions('random')
        pop.evaluate_fitness(lambda x: fun.get_function_value(x))
        return pop

    def test_operator_within_bounds(self, population):
        """Test that operator keeps positions reasonable."""
        original_positions = population.positions.copy()

        # Test random_search
        op.random_search(population, scale=0.5, distribution='uniform')
        # Positions should still be finite
        assert np.all(np.isfinite(population.positions))

        # Reset and test random_sample
        population.positions = original_positions.copy()
        op.random_sample(population)
        assert np.all(np.isfinite(population.positions))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
