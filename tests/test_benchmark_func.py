"""
Test suite for benchmark_func module.
Tests benchmark function creation, evaluation, and properties.
"""

import numpy as np
import pytest

from customhys import benchmark_func as bf


class TestBenchmarkFunctions:
    """Test benchmark function implementations."""

    def test_all_functions_listed(self):
        """Test that __all__ contains function names."""
        assert hasattr(bf, "__all__")
        assert len(bf.__all__) > 0
        assert "Sphere" in bf.__all__
        assert "Rastrigin" in bf.__all__
        assert "Rosenbrock" in bf.__all__

    @pytest.mark.parametrize("dimensions", [2, 5, 10])
    def test_sphere_function_creation(self, dimensions):
        """Test Sphere function creation with different dimensions."""
        fun = bf.Sphere(dimensions)
        assert fun.variable_num == dimensions
        assert fun.func_name == "Sphere"

    def test_sphere_function_evaluation(self):
        """Test Sphere function evaluation."""
        fun = bf.Sphere(2)

        # Test at origin (should be optimal)
        result = fun.get_function_value(np.array([0.0, 0.0]))
        assert np.isclose(result, 0.0, atol=1e-10)

        # Test at non-zero point
        result = fun.get_function_value(np.array([1.0, 1.0]))
        assert result > 0.0

    def test_rastrigin_function(self):
        """Test Rastrigin function."""
        fun = bf.Rastrigin(2)
        assert fun.variable_num == 2

        # Minimum at origin
        result = fun.get_function_value(np.array([0.0, 0.0]))
        assert np.isclose(result, 0.0, atol=1e-10)

    def test_rosenbrock_function(self):
        """Test Rosenbrock function."""
        fun = bf.Rosenbrock(2)
        assert fun.variable_num == 2

        # Minimum at [1, 1]
        result = fun.get_function_value(np.array([1.0, 1.0]))
        assert np.isclose(result, 0.0, atol=1e-10)

    def test_ackley_function(self):
        """Test Ackley function."""
        fun = bf.Ackley1(2)
        assert fun.variable_num == 2

        # Minimum at origin
        result = fun.get_function_value(np.array([0.0, 0.0]))
        assert np.isclose(result, 0.0, atol=1e-10)

    def test_griewank_function(self):
        """Test Griewank function."""
        fun = bf.Griewank(2)
        assert fun.variable_num == 2

        # Minimum at origin
        result = fun.get_function_value(np.array([0.0, 0.0]))
        assert np.isclose(result, 0.0, atol=1e-10)


class TestBenchmarkFunctionProperties:
    """Test benchmark function properties and methods."""

    def test_get_search_range(self):
        """Test search range retrieval."""
        fun = bf.Sphere(2)
        search_range = fun.get_search_range()

        # search_range is typically a tuple of (min_array, max_array)
        assert len(search_range) == 2
        # First element is min bounds, second is max bounds
        if isinstance(search_range[0], np.ndarray):
            assert len(search_range[0]) == 2
            assert len(search_range[1]) == 2
            assert np.all(search_range[0] <= search_range[1])
        else:
            # Or it might be list of [min, max] pairs
            assert len(search_range[0]) == 2
            assert search_range[0][0] <= search_range[0][1]

    def test_get_optimal_fitness(self):
        """Test optimal fitness retrieval."""
        fun = bf.Sphere(2)
        optimal = fun.get_optimal_fitness()
        assert optimal == 0.0

    def test_get_optimal_solution(self):
        """Test optimal solution retrieval."""
        fun = bf.Sphere(2)
        optimal_solution = fun.get_optimal_solution()
        assert len(optimal_solution) == 2
        assert np.allclose(optimal_solution, [0.0, 0.0])

    def test_get_features(self):
        """Test feature retrieval."""
        fun = bf.Sphere(2)
        features = fun.get_features()
        assert isinstance(features, str)
        assert len(features) > 0

    def test_get_formatted_problem(self):
        """Test problem formatting."""
        fun = bf.Sphere(2)
        problem = fun.get_formatted_problem()

        assert "function" in problem
        assert "boundaries" in problem
        assert callable(problem["function"])
        assert len(problem["boundaries"]) == 2


class TestBenchmarkFunctionValues:
    """Test function value calculations."""

    def test_function_values_batch(self):
        """Test batch evaluation of function values."""
        fun = bf.Sphere(2)
        samples = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        values = fun.get_function_values(samples)
        assert len(values) == 3
        assert values[0] < values[1] < values[2]

    def test_function_minimum_verification(self):
        """Test that minimum is at the expected location."""
        fun = bf.Sphere(3)

        # Generate random points
        np.random.seed(42)
        random_points = np.random.uniform(-5, 5, (10, 3))

        # Optimal point
        optimal_point = np.array([0.0, 0.0, 0.0])

        # Optimal should be better than random points
        optimal_value = fun.get_function_value(optimal_point)
        for point in random_points:
            assert optimal_value <= fun.get_function_value(point)

    def test_function_symmetry(self):
        """Test function symmetry for symmetric functions."""
        fun = bf.Sphere(2)

        # Sphere function should be symmetric
        point1 = np.array([1.0, 2.0])
        point2 = np.array([-1.0, -2.0])

        value1 = fun.get_function_value(point1)
        value2 = fun.get_function_value(point2)

        assert np.isclose(value1, value2)


class TestMultipleBenchmarkFunctions:
    """Test multiple benchmark functions."""

    @pytest.mark.parametrize(
        "func_name", ["Sphere", "Rastrigin", "Rosenbrock", "Ackley1", "Griewank", "Schwefel", "Levy"]
    )
    def test_function_instantiation(self, func_name):
        """Test that common functions can be instantiated."""
        if hasattr(bf, func_name):
            func_class = getattr(bf, func_name)
            fun = func_class(2)
            assert fun.variable_num == 2
            # Function names might have spaces, just check it exists and is not empty
            assert len(fun.func_name) > 0

    @pytest.mark.parametrize("func_name", ["Sphere", "Rastrigin", "Rosenbrock"])
    def test_function_evaluation_no_errors(self, func_name):
        """Test that functions can be evaluated without errors."""
        func_class = getattr(bf, func_name)
        fun = func_class(2)

        test_point = np.array([0.5, 0.5])
        result = fun.get_function_value(test_point)

        assert isinstance(result, (int, float, np.number))
        assert np.isfinite(result)


class TestBenchmarkFunctionEdgeCases:
    """Test edge cases and error handling."""

    def test_single_dimension(self):
        """Test function with single dimension."""
        fun = bf.Sphere(1)
        assert fun.variable_num == 1

        result = fun.get_function_value(np.array([0.0]))
        assert np.isclose(result, 0.0)

    def test_high_dimension(self):
        """Test function with high dimensions."""
        fun = bf.Sphere(100)
        assert fun.variable_num == 100

        point = np.zeros(100)
        result = fun.get_function_value(point)
        assert np.isclose(result, 0.0)

    def test_boundary_values(self):
        """Test function evaluation at boundaries."""
        fun = bf.Sphere(2)
        search_range = fun.get_search_range()

        # Test at minimum boundary
        min_point = np.array([r[0] for r in search_range])
        result_min = fun.get_function_value(min_point)
        assert np.isfinite(result_min)

        # Test at maximum boundary
        max_point = np.array([r[1] for r in search_range])
        result_max = fun.get_function_value(max_point)
        assert np.isfinite(result_max)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
