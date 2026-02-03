"""
Test suite for metaheuristic module.
Tests metaheuristic creation, execution, and optimization.
"""

import numpy as np
import pytest

from customhys import benchmark_func as bf
from customhys import metaheuristic as mh


class TestMetaheuristicCreation:
    """Test metaheuristic object creation."""

    def test_metaheuristic_instantiation(self):
        """Test basic metaheuristic creation."""
        fun = bf.Sphere(2)

        search_operators = [("random_search", {"scale": 1.0, "distribution": "uniform"}, "greedy")]

        meta = mh.Metaheuristic(
            problem=fun.get_formatted_problem(),
            search_operators=search_operators,
            num_agents=10,
            num_iterations=5,
            verbose=False,
        )

        assert meta is not None
        assert meta.num_agents == 10
        assert meta.num_iterations == 5

    def test_metaheuristic_with_multiple_operators(self):
        """Test metaheuristic with multiple operators."""
        fun = bf.Rastrigin(2)

        search_operators = [
            ("random_search", {"scale": 1.0, "distribution": "uniform"}, "greedy"),
            ("local_random_walk", {"probability": 0.75, "scale": 1.0, "distribution": "gaussian"}, "greedy"),
        ]

        meta = mh.Metaheuristic(
            problem=fun.get_formatted_problem(),
            search_operators=search_operators,
            num_agents=10,
            num_iterations=5,
            verbose=False,
        )

        # Operators are stored in perturbators and selectors
        assert len(meta.perturbators) == 2

    def test_metaheuristic_different_dimensions(self):
        """Test metaheuristic with different problem dimensions."""
        dimensions = [2, 5, 10]

        for dim in dimensions:
            fun = bf.Sphere(dim)
            search_operators = [("random_search", {"scale": 1.0, "distribution": "uniform"}, "greedy")]

            meta = mh.Metaheuristic(
                problem=fun.get_formatted_problem(),
                search_operators=search_operators,
                num_agents=10,
                num_iterations=5,
                verbose=False,
            )

            assert meta.pop.num_dimensions == dim


class TestMetaheuristicExecution:
    """Test metaheuristic execution and optimization."""

    def test_metaheuristic_run(self):
        """Test running a metaheuristic."""
        fun = bf.Sphere(2)

        search_operators = [("random_search", {"scale": 1.0, "distribution": "uniform"}, "greedy")]

        meta = mh.Metaheuristic(
            problem=fun.get_formatted_problem(),
            search_operators=search_operators,
            num_agents=10,
            num_iterations=10,
            verbose=False,
        )

        meta.run()

        # Check that historical data was recorded
        assert "fitness" in meta.historical
        assert len(meta.historical["fitness"]) > 0

    def test_metaheuristic_optimization_progress(self):
        """Test that optimization makes progress."""
        fun = bf.Sphere(2)

        search_operators = [("random_search", {"scale": 1.0, "distribution": "uniform"}, "greedy")]

        meta = mh.Metaheuristic(
            problem=fun.get_formatted_problem(),
            search_operators=search_operators,
            num_agents=20,
            num_iterations=50,
            verbose=False,
        )

        meta.run()

        # First fitness should be worse than last (for minimization)
        first_fitness = meta.historical["fitness"][0]
        last_fitness = meta.historical["fitness"][-1]

        assert last_fitness <= first_fitness

    def test_metaheuristic_result_format(self):
        """Test that results are in correct format."""
        fun = bf.Sphere(2)

        search_operators = [("random_search", {"scale": 1.0, "distribution": "uniform"}, "greedy")]

        meta = mh.Metaheuristic(
            problem=fun.get_formatted_problem(),
            search_operators=search_operators,
            num_agents=10,
            num_iterations=10,
            verbose=False,
        )

        meta.run()
        position, fitness = meta.get_solution()

        assert len(position) == 2
        # Fitness can be a numpy array or scalar
        if isinstance(fitness, np.ndarray):
            assert fitness.size == 1
            assert np.isfinite(fitness)
        else:
            assert isinstance(fitness, (int, float, np.number))
            assert np.isfinite(fitness)


class TestMetaheuristicWithDifferentFunctions:
    """Test metaheuristic with different benchmark functions."""

    @pytest.mark.parametrize(
        "func_class",
        [
            bf.Sphere,
            bf.Rastrigin,
            bf.Rosenbrock,
        ],
    )
    def test_metaheuristic_different_functions(self, func_class):
        """Test metaheuristic with different functions."""
        fun = func_class(2)

        search_operators = [("random_search", {"scale": 1.0, "distribution": "uniform"}, "greedy")]

        meta = mh.Metaheuristic(
            problem=fun.get_formatted_problem(),
            search_operators=search_operators,
            num_agents=10,
            num_iterations=10,
            verbose=False,
        )

        meta.run()
        position, fitness = meta.get_solution()

        assert len(position) == 2
        assert np.isfinite(fitness)


class TestMetaheuristicSearchOperators:
    """Test metaheuristic with different search operators."""

    @pytest.mark.parametrize(
        "operator",
        [
            ("random_search", {"scale": 1.0, "distribution": "uniform"}, "greedy"),
            ("random_sample", {}, "greedy"),
            ("local_random_walk", {"probability": 0.75, "scale": 1.0, "distribution": "uniform"}, "greedy"),
        ],
    )
    def test_metaheuristic_with_operator(self, operator):
        """Test metaheuristic with different operators."""
        fun = bf.Sphere(2)

        meta = mh.Metaheuristic(
            problem=fun.get_formatted_problem(),
            search_operators=[operator],
            num_agents=10,
            num_iterations=10,
            verbose=False,
        )

        meta.run()
        position, fitness = meta.get_solution()

        assert len(position) == 2
        assert np.isfinite(fitness)


class TestMetaheuristicSelectors:
    """Test metaheuristic with different selectors."""

    @pytest.mark.parametrize("selector", ["greedy", "all", "metropolis", "probabilistic"])
    def test_metaheuristic_selectors(self, selector):
        """Test metaheuristic with different selection methods."""
        fun = bf.Sphere(2)

        search_operators = [("random_search", {"scale": 1.0, "distribution": "uniform"}, selector)]

        meta = mh.Metaheuristic(
            problem=fun.get_formatted_problem(),
            search_operators=search_operators,
            num_agents=10,
            num_iterations=10,
            verbose=False,
        )

        meta.run()
        position, fitness = meta.get_solution()

        assert len(position) == 2
        assert np.isfinite(fitness)


class TestMetaheuristicHistorical:
    """Test metaheuristic historical data tracking."""

    def test_historical_fitness_recorded(self):
        """Test that fitness history is recorded."""
        fun = bf.Sphere(2)

        search_operators = [("random_search", {"scale": 1.0, "distribution": "uniform"}, "greedy")]

        num_iterations = 20

        meta = mh.Metaheuristic(
            problem=fun.get_formatted_problem(),
            search_operators=search_operators,
            num_agents=10,
            num_iterations=num_iterations,
            verbose=False,
        )

        meta.run()

        # Should have num_iterations + 1 records (initial + iterations)
        assert len(meta.historical["fitness"]) == num_iterations + 1

    def test_historical_position_recorded(self):
        """Test that position history is recorded."""
        fun = bf.Sphere(2)

        search_operators = [("random_search", {"scale": 1.0, "distribution": "uniform"}, "greedy")]

        meta = mh.Metaheuristic(
            problem=fun.get_formatted_problem(),
            search_operators=search_operators,
            num_agents=10,
            num_iterations=10,
            verbose=False,
        )

        meta.run()

        assert "position" in meta.historical
        assert len(meta.historical["position"]) > 0
        assert len(meta.historical["position"][0]) == 2

    def test_historical_data_consistency(self):
        """Test that historical data is consistent."""
        fun = bf.Sphere(2)

        search_operators = [("random_search", {"scale": 1.0, "distribution": "uniform"}, "greedy")]

        meta = mh.Metaheuristic(
            problem=fun.get_formatted_problem(),
            search_operators=search_operators,
            num_agents=10,
            num_iterations=10,
            verbose=False,
        )

        meta.run()

        # All historical arrays should have same length
        assert len(meta.historical["fitness"]) == len(meta.historical["position"])


class TestMetaheuristicPopulationSize:
    """Test metaheuristic with different population sizes."""

    @pytest.mark.parametrize("num_agents", [5, 10, 20, 50])
    def test_different_population_sizes(self, num_agents):
        """Test metaheuristic with different population sizes."""
        fun = bf.Sphere(2)

        search_operators = [("random_search", {"scale": 1.0, "distribution": "uniform"}, "greedy")]

        meta = mh.Metaheuristic(
            problem=fun.get_formatted_problem(),
            search_operators=search_operators,
            num_agents=num_agents,
            num_iterations=10,
            verbose=False,
        )

        meta.run()
        position, fitness = meta.get_solution()

        assert len(position) == 2
        assert np.isfinite(fitness)


class TestMetaheuristicInitialization:
    """Test metaheuristic initialization schemes."""

    @pytest.mark.parametrize("scheme", ["random", "vertex"])
    def test_initialization_schemes(self, scheme):
        """Test different initialization schemes."""
        fun = bf.Sphere(2)

        search_operators = [("random_search", {"scale": 1.0, "distribution": "uniform"}, "greedy")]

        meta = mh.Metaheuristic(
            problem=fun.get_formatted_problem(),
            search_operators=search_operators,
            num_agents=10,
            num_iterations=10,
            initial_scheme=scheme,
            verbose=False,
        )

        meta.run()
        position, fitness = meta.get_solution()

        assert len(position) == 2
        assert np.isfinite(fitness)


class TestMetaheuristicConvergence:
    """Test metaheuristic convergence behavior."""

    def test_convergence_to_optimum(self):
        """Test that metaheuristic converges towards optimum."""
        fun = bf.Sphere(2)

        search_operators = [
            ("random_search", {"scale": 1.0, "distribution": "uniform"}, "greedy"),
            ("local_random_walk", {"probability": 0.75, "scale": 0.5, "distribution": "gaussian"}, "greedy"),
        ]

        meta = mh.Metaheuristic(
            problem=fun.get_formatted_problem(),
            search_operators=search_operators,
            num_agents=30,
            num_iterations=100,
            verbose=False,
        )

        meta.run()
        position, fitness = meta.get_solution()

        # For Sphere function, optimum is at origin with fitness 0
        # Random initialization in [-100, 100] range can give initial fitness ~20000
        # After 100 iterations, should show significant improvement
        fitness_value = float(fitness) if isinstance(fitness, np.ndarray) else fitness
        assert fitness_value < 200.0  # Should improve significantly from random initialization
        assert np.isfinite(fitness_value)  # Should be a valid number


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
