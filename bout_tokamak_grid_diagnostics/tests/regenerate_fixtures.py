#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
import argparse
import os
import numpy as np
from netCDF4 import Dataset

# ==============================================================================
# Chatwood Labs – Maintainer Utility: Test Fixture Regenerator
#
# regenerate_fixtures.py
#
# Internal helper script for regenerating the small netCDF grid fixtures used by
# the bout_tokamak_grid_diagnostics test suite.
#
# This script:
#   - invokes the external bout_tokamak_grid_generator.py tool
#   - generates a small "good" reference grid
#   - deliberately mutates copies to create broken grids (Jacobian sign flip,
#     NaN in metric tensor, etc.) for negative test coverage
#
# It is NOT part of the public diagnostics tool.
# It is NOT required for normal usage.
# It is NOT safe to run unless you understand what it does.
#
# This exists purely to keep test data:
#   - deterministic
#   - reproducible
#   - and under version control
#
# Design note:
# This script is intentionally kept as a single file and placed under tests/
# to make its role and scope explicit. It depends on an external generator
# and will fail loudly if that dependency is missing.
#
# License:
#   Released under the MIT License (see repository LICENSE file).
#
# © 2025 Chatwood Labs Ltd
# ==============================================================================

def tests_dir() -> Path:
    # .../bout_tokamak_grid_diagnostics/tests/
    return Path(__file__).resolve().parent

def diagnostics_pkg_dir() -> Path:
    # .../bout_tokamak_grid_diagnostics/
    return tests_dir().parent

def repo_root() -> Path:
    # monorepo root (parent of bout_tokamak_grid_diagnostics)
    return diagnostics_pkg_dir().parent

def generator_path(cli_override: Path | None = None) -> Path:
    """
    Resolve the grid generator script path.

    Priority:
      1) CLI override (--generator)
      2) Env var BOUT_TOKAMAK_GRID_GENERATOR
      3) Monorepo default: <repo_root>/bout_tokamak_grid/bout_tokamak_grid_generator.py

    Notes:
      - We *always* normalize to an absolute resolved Path when possible.
      - If the user passes a directory to --generator by mistake, we treat it as
        "that directory contains bout_tokamak_grid_generator.py" (more forgiving, less dumb).
    """
    if cli_override is not None:
        p = Path(cli_override).expanduser()
        # Forgive the common mistake: --generator /path/to/bout_tokamak_grid/
        if p.is_dir():
            p = p / "bout_tokamak_grid_generator.py"
        return p.resolve()

    env = os.environ.get("BOUT_TOKAMAK_GRID_GENERATOR", "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_dir():
            p = p / "bout_tokamak_grid_generator.py"
        return p.resolve()

    # Default: bout-tools monorepo layout
    return repo_root() / "bout_tokamak_grid" / "bout_tokamak_grid_generator.py"

# Run external commands with captured stdout/stderr so failures are actionable in CI logs.
def run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            f"  {' '.join(cmd)}\n\n"
            f"STDOUT:\n{r.stdout}\n\n"
            f"STDERR:\n{r.stderr}\n"
        )

def generate_grid(gen_py: Path, outfile: Path, *, nx=20, ny=32, nz=8, kappa=1.0, delta=0.0) -> None:
    outfile.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(gen_py),
        "--outfile", str(outfile),
        "--nx", str(nx), "--ny", str(ny), "--nz", str(nz),
        "--kappa", str(kappa), "--delta", str(delta),
        "--qform", "quadratic", "--q0", "1.05", "--qa", "3.5",
        "--curvature", "simple",
    ]
    run(cmd)

def copy_and_mutate(good: Path, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(good, out)
    except PermissionError:
        shutil.copyfile(good, out)

def make_bad_j_signflip(good: Path, out: Path) -> None:
    copy_and_mutate(good, out)
    with Dataset(out, "r+") as nc:
        if "J" not in nc.variables:
            raise KeyError(f"{out.name}: missing variable 'J' to mutate")
        J = nc.variables["J"][:]
        if J.ndim != 3:
            raise ValueError(f"{out.name}: expected J to be 3D, got shape {J.shape}")
        ny = J.shape[1]
        J[:, ny // 2 :, :] *= -1.0
        nc.variables["J"][:] = J

        # Keep derived volume element consistent after we intentionally corrupt J.
        # Some grids store surfvol explicitly (often ~ J*dx*dy*dz). If we flip the
        # Jacobian sign but leave surfvol unchanged, diagnostics might report
        # confusing mixed signals (J negative but surfvol still positive).
        if all(v in nc.variables for v in ("surfvol", "dx", "dy", "dz")):
            dx = nc.variables["dx"][:]
            dy = nc.variables["dy"][:]
            dz = nc.variables["dz"][:]
            nc.variables["surfvol"][:] = J * dx * dy * dz

def make_bad_g11_nan(good: Path, out: Path) -> None:
    copy_and_mutate(good, out)
    with Dataset(out, "r+") as nc:
        if "g11" not in nc.variables:
            raise KeyError(f"{out.name}: missing variable 'g11' to mutate")
        g11 = nc.variables["g11"][:]
        g11_flat = g11.reshape(-1)
        g11_flat[0] = np.nan
        nc.variables["g11"][:] = g11_flat.reshape(g11.shape)

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate test grid fixtures for bout_tokamak_grid_diagnostics."
    )
    parser.add_argument(
        "--generator",
        type=Path,
        default=None,
        help="Path to bout_tokamak_grid_generator.py (overrides env + default).",
    )
    args = parser.parse_args(argv)

    gen_script = generator_path(args.generator)

    # This script is a *maintainer* helper. It is not required to run the diagnostics tool.
    # It regenerates the tiny netCDF fixtures used by the unit tests, and it depends on the
    # separate grid generator script.
    #
    # By design:
    # - If the generator cannot be found, we fail with exit code 2 (usage/config error).
    # - The user can fix it by providing --generator, or by setting BOUT_TOKAMAK_GRID_GENERATOR.
    if not gen_script.exists():
        default_guess = (
            repo_root()
            / "bout_tokamak_grid"
            / "bout_tokamak_grid_generator.py"
        )
        env_hint = os.environ.get("BOUT_TOKAMAK_GRID_GENERATOR", "").strip()

        print("ERROR: required grid generator script was not found.")
        print(f"Resolved path: {gen_script}")

        if args.generator is not None:
            print("Cause: you provided --generator, but that path does not exist.")
        elif env_hint:
            print("Cause: BOUT_TOKAMAK_GRID_GENERATOR is set, but it points to a missing file.")
            print(f"BOUT_TOKAMAK_GRID_GENERATOR={env_hint}")
        else:
            print("Cause: no --generator and no BOUT_TOKAMAK_GRID_GENERATOR set.")
            print(f"Expected default location (monorepo layout): {default_guess}")

        print("")
        print("Fix options:")
        print("  1) Pass the generator explicitly:")
        print("     python tests/regenerate_fixtures.py --generator /path/to/bout_tokamak_grid_generator.py")
        print("")
        print("  2) Or set an environment variable (Linux/macOS):")
        print("     export BOUT_TOKAMAK_GRID_GENERATOR=/path/to/bout_tokamak_grid_generator.py")
        print("")
        print("  3) Or set an environment variable (Windows PowerShell):")
        print("     setx BOUT_TOKAMAK_GRID_GENERATOR \"C:\\path\\to\\bout_tokamak_grid_generator.py\"")
        print("")
        return 2

    # ---------------------------------------------------------------------
    # Fixture generation (this is the whole reason this script exists)
    #
    # Tests expect grids in:
    #   tests/data/grids/
    #
    # Expected fixture filenames (see tests/test_cli.py):
    #   - grid_tiny_shaped.nc
    #   - grid_tiny_bad_J_signflip.nc
    #   - grid_tiny_bad_g11_nan.nc
    #
    # We generate the good grid using the external generator, then mutate copies
    # to create deterministic "bad" fixtures for critical failure tests.
    # ---------------------------------------------------------------------
    grids_out_dir = tests_dir() / "data" / "grids"
    grids_out_dir.mkdir(parents=True, exist_ok=True)

    good_grid = grids_out_dir / "grid_tiny_shaped.nc"
    bad_j_grid = grids_out_dir / "grid_tiny_bad_J_signflip.nc"
    bad_nan_grid = grids_out_dir / "grid_tiny_bad_g11_nan.nc"

    # 1) Generate the baseline "good" grid.
    #
    # Keep dimensions small-ish so fixtures stay lightweight in git,
    # but not so tiny they become numerically degenerate.
    #
    # NOTE: ny=32 matches diagnostics warning threshold default
    # (--ny-warn-min=32). We don't want fixtures to randomly trip warnings
    # just because the default policy changed.
    print(f"[regen] generating good fixture: {good_grid.name}")
    generate_grid(
        gen_script,
        good_grid,
        nx=20,
        ny=32,
        nz=8,
        kappa=1.4,
        delta=0.25,
    )

    # 2) Create a Jacobian sign-flip mutant (critical failure).
    print(f"[regen] generating bad fixture (J sign flip): {bad_j_grid.name}")
    make_bad_j_signflip(good_grid, bad_j_grid)

    # 3) Create a NaN metric mutant (critical failure).
    print(f"[regen] generating bad fixture (g11 NaN): {bad_nan_grid.name}")
    make_bad_g11_nan(good_grid, bad_nan_grid)

    print("")
    print("[regen] done.")
    print(f"[regen] output dir: {grids_out_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())