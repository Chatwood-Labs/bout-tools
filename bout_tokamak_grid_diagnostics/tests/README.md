# Test Suite - bout_tokamak_grid_diagnostics

This directory contains the automated test suite for the BOUT++ Tokamak Grid Diagnostics utility.

The tests are intentionally **contract-focused** they validate CLI behaviour, exit codes, output generation, and failure detection - not just internal functions.

The goal is to ensure that:
- real users invoking the CLI get predictable results
- critical grid pathologies are correctly detected
- regressions are caught early in CI

## Directory Structure

- tests/
	- grids/
		- grid_tiny_circular.nc
		- grid_tiny_bad_J_signflip.nc
		- grid_tiny_bad_g11_nan.nc
	- conftest.py
	- regenerate_fixtures.py
	- test_cli.py

## `data/grids/` - Test Fixtures

This directory contains **small, deterministic netCDF grids** used as fixtures for testing.

They are intentionally tiny to keep:
- repository size small
- test execution fast
- CI stable and repeatable

### Fixtures included

| File | Purpose |
|------|---------|
| `grid_tiny_circular.nc` | Simple circular geometry. Used for baseline sanity tests. |
| `grid_tiny_bad_J_signflip.nc` | Intentionally corrupted Jacobian (sign flip). Must fail critical checks. |
| `grid_tiny_bad_g11_nan.nc` | Intentionally injected NaN in metric tensor. Must fail critical checks. |

The "bad" grids are **deliberately broken**.  
They exist to verify that the diagnostics tool correctly detects and reports fatal geometry errors.

Do not "fix" these files.

## `test_cli.py`

This file tests the tool **as a user would run it**:

- invokes the CLI via subprocess
- checks exit codes
- validates that reports are generated
- verifies that critical failures are correctly detected

These are not unit tests. They are **behavioural contract tests**.

If these fail, it means the user-facing behaviour has changed.

## `conftest.py`

Shared pytest fixtures and helpers.

Primarily responsible for:

- resolving paths to test grids
- loading JSON outputs for inspection

Kept minimal by design.

## `regenerate_fixtures.py`

Maintainer helper script.

This script regenerates the fixtures in `data/grids/` using the external
`bout_tokamak_grid_generator.py` tool.

It is **not required** for normal use or for running the tests.

It exists so that:
- fixture generation is reproducible
- future changes to the generator can be reflected cleanly in test data

Usage:

```bash
python tests/regenerate_fixtures.py --generator /path/to/bout_tokamak_grid_generator.py
```

Do not run this unless you know why you are doing it.

## Design Philosophy

The tests in this directory are intentionally:

- small
- explicit
- deterministic

They are designed to catch:

- geometry corruption
- metric tensor failures
- Jacobian sign errors
- CLI regressions

They are **not** intended to validate physical equilibria or scientific correctness. That is the job of upstream grid generation and equilibrium tools.

This suite validates the **diagnostics tool itself**.