"""Execute the examples in fresh kernels with no local collections or saved runs.

Install nbclient, nbformat and ipykernel to include these integration checks.
The notebook CI job installs them explicitly; core-only environments skip them.
"""

import os
import sys
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")
pytest.importorskip("ipykernel")
KernelManager = pytest.importorskip("jupyter_client").KernelManager

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "examples").glob("*.ipynb"))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.stem)
def test_notebook_runs_from_clean_directory(path, tmp_path):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    # Use the test environment, even if another Python owns the python3 kernelspec.
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), env.get("PYTHONPATH")]))
    client = nbclient.NotebookClient(
        notebook,
        km=manager,
        timeout=180,
        resources={"metadata": {"path": str(tmp_path)}},
    )
    # Check numerical behaviour inside the same kernel, after all example cells.
    checks = """
assert len(met.historical['fitness']) == 1 + met.num_iterations * len(met.perturbators)
assert np.all(np.isfinite(met.historical['fitness']))
assert np.all(np.diff(met.historical['fitness']) <= 1e-10)
assert np.all(np.isfinite(history))
assert np.all(np.diff(best_history) <= 1e-10)
"""
    if path.stem.startswith("Tutorial_"):
        checks += """
assert len(full_data) >= 1
assert np.isfinite(best_perf)
assert np.all(np.diff(hist_best) <= 1e-10)
np.testing.assert_allclose(full_data[-1]['performance'], best_perf)
assert best_sequence[0][0].startswith('initialize_')
assert discovered_operators == best_sequence[1:]
assert initial_scheme == (best_sequence[0][0], best_sequence[0][1])
assert len(hybrid_met.perturbators) == len(hybrid_sequence)
assert fitness_array.shape == (1 + NUM_ITERATIONS * len(hybrid_sequence), NUM_REPLICAS)
assert len(discovered_met.historical['fitness']) == 1 + parameters['num_iterations'] * len(discovered_operators)
assert np.all(np.isfinite(cec_met.historical['fitness']))
assert np.isfinite(cec_perf)
assert len(initial_samples) == 11
for sample in initial_samples.values():
    assert sample.shape == (INITIAL_AGENTS, 2)
    assert np.all(np.isfinite(sample))
    assert np.all((sample >= -5) & (sample <= 5))
lhs_intervals = np.floor((initial_samples['lhs'] + 5) / 10 * INITIAL_AGENTS).astype(int)
for axis in range(2):
    assert len(np.unique(lhs_intervals[:, axis])) == INITIAL_AGENTS
assert rosenbrock([1, 1]) == 0
assert rosenbrock([0, 0]) == 1
assert np.isfinite(rosen_perf)
assert np.all(np.diff(rosen_best) <= 1e-10)
np.testing.assert_allclose(rosenbrock(rosen_position), rosen_fitness, atol=1e-10)
np.random.seed(SEED + 1)
initial_rosen = pp.Population(rosen_problem['boundaries'], num_agents=NUM_AGENTS)
initial_rosen.initialise_positions('lhs')
np.testing.assert_allclose(rosen_run.population_history[0], initial_rosen.get_positions())
for run, objective in [(trace_rastrigin, visual_fun.get_function_value),
                       (discovered_met, hh_fun.get_function_value),
                       (rosen_run, rosenbrock)]:
    trace = np.asarray(run.population_history)
    assert trace.shape == (len(run.historical['fitness']), run.num_agents, 2)
    assert np.all(trace >= run.pop.lower_boundaries - 1e-12)
    assert np.all(trace <= run.pop.upper_boundaries + 1e-12)
    np.testing.assert_allclose(trace[-1], run.pop.get_positions())
    evaluated = [[objective(point) for point in snapshot] for snapshot in trace]
    np.testing.assert_allclose(evaluated, run.population_fitness, atol=1e-10)
    best_points = run.pop.rescale_back(np.asarray(run.historical['position']))
    np.testing.assert_allclose([objective(point) for point in best_points], run.historical['fitness'], atol=1e-10)
assert frame_indices[0] == 0
assert frame_indices[-1] == len(rosen_run.population_history) - 1
assert 'data:image/png;base64,' in animation_html
"""
    else:
        checks += """
assert positions.shape == (NUM_ITERATIONS + 1, NUM_AGENTS, NUM_DIMENSIONS)
assert mh_matrices.shape == (NUM_AGENTS, met.num_dimensions, met.num_dimensions)
np.testing.assert_allclose(fun.get_function_value(best_positions[-1]), best_history[-1])
"""
    notebook.cells.append(nbformat.v4.new_code_cell(checks))
    executed = client.execute(env=env, cleanup_kc=True)
    code_cells = [cell for cell in executed.cells if cell.cell_type == "code"]
    assert all(cell.execution_count is not None for cell in code_cells)
    assert any(
        "image/png" in output.get("data", {}) for cell in code_cells for output in cell.outputs
    ), "The example must render its figures without an external LaTeX installation"
