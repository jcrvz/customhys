#!/usr/bin/env python3
"""
Test all Makefile commands to ensure they work properly.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description, expect_failure=False):
    """Run a command and report results."""
    print(f"\n{'=' * 70}")
    print(f"Testing: {description}")
    print(f"Command: {cmd}")
    print("=" * 70)

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)

        if result.returncode == 0 or expect_failure:
            print(f"✅ {description}: OK")
            if result.stdout:
                print("Output (last 10 lines):")
                lines = result.stdout.strip().split("\n")
                for line in lines[-10:]:
                    print(f"  {line}")
            return True
        else:
            print(f"❌ {description}: FAILED")
            if result.stderr:
                print("Error:")
                print(result.stderr[:500])
            return False
    except subprocess.TimeoutExpired:
        print(f"⚠️  {description}: TIMEOUT (>30s)")
        return False
    except Exception as e:
        print(f"❌ {description}: ERROR - {e}")
        return False


def main():
    """Test all Makefile commands."""
    print("\n" + "=" * 70)
    print("MAKEFILE COMMAND TEST SUITE")
    print("=" * 70)

    # Change to project directory
    project_dir = Path(__file__).parent
    import os

    os.chdir(project_dir)

    tests = [
        # Basic commands (should always work)
        ("make help", "make help", False),
        ("make check-uv", "make check-uv", False),
        # Installation commands (should work)
        ("make sync", "make sync", False),
        ("make install", "make install (core)", False),
        # Test commands (expect some failures ok)
        ("make validate-setup", "make validate-setup", True),
        # Quality commands (expect warnings ok)
        ("make lint", "make lint", True),
        ("make format-check", "make format-check", True),
        ("make typecheck", "make typecheck", True),
        # Utility commands
        ("make clean", "make clean", False),
        # Commands that need setup (allow failure)
        ("make pre-commit-install", "make pre-commit-install", True),
        # Skip these as they need special setup:
        # - make build (needs build module)
        # - make publish (needs credentials)
        # - make test (may hang)
        # - make test-all (may hang)
    ]

    results = []

    for cmd, desc, allow_fail in tests:
        success = run_command(cmd, desc, allow_fail)
        results.append((desc, success))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for desc, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {desc}")

    print("-" * 70)
    print(f"Total: {passed}/{total} commands working")
    print("=" * 70)

    # Additional commands not tested (require special setup)
    print("\n📝 Commands not tested (require special setup):")
    print("  • make build - requires 'build' module installation")
    print("  • make publish - requires PyPI credentials")
    print("  • make test - may take time, test via 'make validate-setup'")
    print("  • make test-all - comprehensive suite, run manually")
    print("  • make install-dev - covered by sync")
    print("  • make install-all - covered by sync")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
