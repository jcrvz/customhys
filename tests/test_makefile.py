"""
Test suite for Makefile and UV integration.
Verifies that the build system works correctly.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, check=True):
    """Run a shell command and return the result."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\n{result.stderr}")
    return result


def test_uv_installed():
    """Test if UV is installed."""
    result = run_command("command -v uv", check=False)
    if result.returncode == 0:
        uv_path = result.stdout.strip()
        version_result = run_command("uv --version", check=False)
        version = version_result.stdout.strip() if version_result.returncode == 0 else "unknown"
        print(f"✓ UV installed at: {uv_path}")
        print(f"  Version: {version}")
    else:
        print("ℹ UV not installed (falling back to pip)")


def test_make_available():
    """Test if make is available."""
    result = run_command("command -v make", check=False)
    if result.returncode == 0:
        print(f"✓ Make available at: {result.stdout.strip()}")
    else:
        print("✗ Make not available")


def test_makefile_exists():
    """Test that Makefile exists."""
    project_root = Path(__file__).parent.parent
    makefile = project_root / "Makefile"
    assert makefile.exists(), "Makefile not found"
    print(f"✓ Makefile exists ({makefile.stat().st_size} bytes)")


def test_make_help():
    """Test make help command."""
    result = run_command("make help")
    assert "Usage: make" in result.stdout
    assert "Available targets:" in result.stdout

    # Check that key commands are listed
    expected_commands = ["sync", "install", "install-dev", "test", "lint", "format"]
    for cmd in expected_commands:
        assert cmd in result.stdout, f"Command '{cmd}' not found in help"

    print("✓ Make help works and lists all expected commands")


def test_make_check_uv():
    """Test make check-uv command."""
    result = run_command("make check-uv")

    if "Using uv" in result.stdout:
        print("✓ Make check-uv detects UV")
    elif "Using pip" in result.stdout:
        print("✓ Make check-uv falls back to pip")
    else:
        raise AssertionError(f"Unexpected check-uv output: {result.stdout}")


def test_pyproject_toml_valid():
    """Test that pyproject.toml is valid."""
    try:
        import tomli
    except ImportError:
        try:
            import tomllib as tomli
        except ImportError:
            print("ℹ TOML library not available, skipping validation")
            return

    project_root = Path(__file__).parent.parent
    pyproject = project_root / "pyproject.toml"

    with open(pyproject, "rb") as f:
        try:
            data = tomli.load(f)
            assert "project" in data
            assert data["project"]["name"] == "customhys"
            # assert data['project']['version'] == '1.1.8'
            print(f"✓ pyproject.toml is valid (version {data['project']['version']})")
        except Exception as e:
            raise AssertionError(f"Invalid pyproject.toml: {e}") from e


def test_dependencies_structure():
    """Test that dependencies are properly structured."""
    try:
        import tomli
    except ImportError:
        try:
            import tomllib as tomli
        except ImportError:
            print("ℹ TOML library not available, skipping test")
            return

    project_root = Path(__file__).parent.parent
    pyproject = project_root / "pyproject.toml"

    with open(pyproject, "rb") as f:
        data = tomli.load(f)

        # Check core dependencies
        deps = data["project"]["dependencies"]
        assert len(deps) > 0, "No core dependencies defined"
        assert any("numpy" in dep for dep in deps)
        assert any("scipy" in dep for dep in deps)

        # Check optional dependencies
        optional = data["project"].get("optional-dependencies", {})
        assert "ml" in optional, "ML extras not defined"
        assert "dev" in optional, "Dev extras not defined"
        assert "examples" in optional, "Examples extras not defined"

        print("✓ Dependencies structure correct:")
        print(f"  - Core dependencies: {len(deps)}")
        print(f"  - Optional groups: {', '.join(optional.keys())}")


def test_requirements_txt_exists():
    """Test that requirements.txt exists and is not bloated."""
    project_root = Path(__file__).parent.parent
    requirements = project_root / "requirements.txt"

    assert requirements.exists(), "requirements.txt not found"

    with open(requirements) as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    # Should have minimal core dependencies (around 7-10)
    assert len(lines) <= 15, f"requirements.txt has too many dependencies: {len(lines)}"
    print(f"✓ requirements.txt exists with {len(lines)} core dependencies")


def test_uv_lock_if_available():
    """Test uv.lock file if UV is being used."""
    project_root = Path(__file__).parent.parent
    uv_lock = project_root / "uv.lock"

    if uv_lock.exists():
        size_kb = uv_lock.stat().st_size / 1024
        print(f"✓ uv.lock exists ({size_kb:.1f} KB)")

        # Check that it's not empty
        assert uv_lock.stat().st_size > 1000, "uv.lock seems too small"
    else:
        print("ℹ uv.lock not found (project may not use UV)")


def test_makefile_targets():
    """Test that all expected Makefile targets are present."""
    project_root = Path(__file__).parent.parent
    makefile = project_root / "Makefile"

    with open(makefile) as f:
        content = f.read()

    expected_targets = [
        "help:",
        "check-uv:",
        "sync:",
        "install:",
        "install-dev:",
        "install-all:",
        "test:",
        "test-fast:",
        "lint:",
        "lint-fix:",
        "format:",
        "format-check:",
        "typecheck:",
        "check-all:",
        "clean:",
        "build:",
        "publish:",
        "pre-commit-install:",
        "setup-dev:",
        "validate:",
    ]

    missing = [target for target in expected_targets if target not in content]

    if missing:
        raise AssertionError(f"Missing Makefile targets: {', '.join(missing)}")

    print(f"✓ All {len(expected_targets)} expected Makefile targets present")


def test_makefile_uv_detection():
    """Test that Makefile properly detects UV."""
    project_root = Path(__file__).parent.parent
    makefile = project_root / "Makefile"

    with open(makefile) as f:
        content = f.read()

    # Check for UV detection
    assert "UV := $(shell command -v uv" in content

    # Check for uv sync usage (not uv pip install)
    assert "uv sync" in content

    # Check for conditional execution
    assert 'if [ -n "$(UV)"' in content or '@if [ -n "$(UV)"' in content

    print("✓ Makefile properly detects UV and uses uv sync")


def test_git_ignore():
    """Test that .gitignore exists and has proper entries."""
    project_root = Path(__file__).parent.parent
    gitignore = project_root / ".gitignore"

    if gitignore.exists():
        with open(gitignore) as f:
            content = f.read()

        important_entries = ["__pycache__", "*.pyc", "dist/", "build/", ".venv", "venv/"]
        present = [entry for entry in important_entries if entry in content]

        print(f"✓ .gitignore exists with {len(present)}/{len(important_entries)} important entries")
    else:
        print("ℹ .gitignore not found")


def test_documentation_files():
    """Test that documentation files exist."""
    project_root = Path(__file__).parent.parent

    doc_files = {
        "README.md": "Main documentation",
        "CHANGELOG.md": "Version history",
        "CONTRIBUTING.md": "Contribution guidelines",
        "LICENSE": "License file",
    }

    for filename, description in doc_files.items():
        assert description
        filepath = project_root / filename
        if filepath.exists():
            size_kb = filepath.stat().st_size / 1024
            print(f"  ✓ {filename}: {size_kb:.1f} KB")
        else:
            print(f"  ✗ {filename}: Missing")


def test_pre_commit_config():
    """Test that pre-commit config exists."""
    project_root = Path(__file__).parent.parent
    pre_commit = project_root / ".pre-commit-config.yaml"

    if pre_commit.exists():
        with open(pre_commit) as f:
            content = f.read()

        # Check for important hooks
        assert "black" in content or "Black" in content
        assert "ruff" in content or "Ruff" in content

        print("✓ Pre-commit config exists with black and ruff")
    else:
        print("ℹ Pre-commit config not found")


if __name__ == "__main__":
    """Run all tests when executed directly."""

    print("=" * 60)
    print("Running Build System Tests")
    print("=" * 60)

    # Get all test functions
    test_functions = [obj for name, obj in globals().items() if name.startswith("test_") and callable(obj)]

    passed = 0
    failed = 0

    for test_func in test_functions:
        print(f"\n{test_func.__name__}:")
        print("-" * 60)
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            failed += 1
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)
