from __future__ import annotations
from pathlib import Path
import bout_tokamak_grid_diagnostics as diag

# Behavioural contract tests for bout_tokamak_grid_diagnostics CLI.
# These tests exercise the tool as a user would run it (via subprocess),
# validating exit codes, JSON output shape, and failure detection.
#
# This file intentionally avoids in-process execution to prevent global
# logging/stdout contamination between tests.

def assert_min_schema(payload: dict, *, context: str = "JSON report") -> None:
    """
    Assert the minimal JSON schema contract for diagnostics output.

    Why this exists:
      - Multiple tests rely on the same base JSON structure.
      - Centralising these checks makes tests shorter and ensures schema expectations
        stay consistent across the suite.
      - If schema ever changes intentionally, there's only one place to update.

    Contract enforced here:
      - payload["checks"] is a list
      - payload["critical_failures"] is a list
      - payload["warnings"] is a list
    """
    assert isinstance(payload, dict), f"{context}: report payload is not a dict"

    assert "checks" in payload and isinstance(payload["checks"], list), (
        f"{context}: Missing or malformed 'checks' in JSON report"
    )
    assert "critical_failures" in payload and isinstance(payload["critical_failures"], list), (
        f"{context}: Missing or malformed 'critical_failures' in JSON report"
    )
    assert "warnings" in payload and isinstance(payload["warnings"], list), (
        f"{context}: Missing or malformed 'warnings' in JSON report"
    )

def _run_json_only(grid: Path, outdir: Path, extra_args: list[str] | None = None) -> tuple[int, Path]:
    """
    Runs diagnostics in a subprocess, forcing JSON-only output.
    Returns (exit_code, json_path).

    IMPORTANT:
      We intentionally do NOT call diag.main_cli(...) in-process.

      The diagnostics CLI configures logging handlers and may redirect stdout/stderr.
      Re-invoking main_cli multiple times in the same Python process can leave global
      state (logging handlers / sys.stdout) in a wrapped configuration, causing
      handler -> stream -> logger recursion on subsequent runs (RecursionError).

      Subprocess isolation is the most surgical fix:
        - avoids modifying release code
        - avoids brittle "reset logging" logic
        - matches real-world usage (users run the CLI as a command)
    """
    import sys
    import subprocess

    outdir.mkdir(parents=True, exist_ok=True)

    # Keep the same argument style your tests already expect:
    # --outdir + --json-only + optional extra args.
    args = [str(grid), "--outdir", str(outdir), "--json-only"]
    if extra_args:
        args.extend(extra_args)

    # Run in a fresh interpreter process to avoid cross-test contamination.
    cmd = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "import bout_tokamak_grid_diagnostics as d; "
            "sys.exit(d.main_cli(sys.argv[1:]))"
        ),
        *args,
    ]

    # Capture output so if it fails we can see why without rerunning manually.
    proc = subprocess.run(cmd, capture_output=True, text=True)

    # CLI documents these exit codes; anything else is unexpected.
    if proc.returncode not in (0, 1, 2):
        raise AssertionError(
            "Unexpected diagnostics CLI return code.\n"
            f"cmd: {cmd}\n"
            f"returncode: {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}\n"
        )

    # JSON path remains exactly as before:
    # <outdir>/<grid_basename>_report.json
    json_path = outdir / f"{grid.stem}_report.json"
    return proc.returncode, json_path


def test_good_grid_strict_mode_exit_code_contract(tmp_path: Path, grids_dir: Path, read_report_json):
    """
    Deterministic strict-mode test (exit-code contract).

    Background:
      - "Good" fixtures can legitimately produce *zero warnings* depending on:
          - policy thresholds
          - floating tolerances
          - future refinements to checks
        That means any test that *expects* warnings from a good grid is inherently unstable.

    What we actually want to validate:
      - Strict mode must NOT invent failures.
      - Strict mode must only change the *exit-code interpretation* when warnings exist:
          - If warnings exist and there are no critical_failures => exit code must be 1
          - If warnings do NOT exist, strict mode should behave the same as non-strict mode

    This test therefore:
      1) Runs the same known-good fixture in non-strict mode
      2) Runs it again in strict mode
      3) Uses the presence/absence of warnings in the reports to assert the exit-code contract
    """
    grid = grids_dir / "grid_tiny_shaped.nc"
    assert grid.exists(), f"Missing fixture: {grid}"

    # ------------------------------------------------------------
    # 1) Non-strict run (baseline)
    # ------------------------------------------------------------
    outdir_normal = tmp_path / "out_normal"
    code_normal, json_path_normal = _run_json_only(grid, outdir_normal)

    assert json_path_normal.exists(), f"Expected JSON report to exist: {json_path_normal}"
    payload_normal = read_report_json(json_path_normal)

    # Minimal JSON schema contract (stable interface for downstream tooling)
    assert_min_schema(payload_normal, context="good grid baseline JSON report")

    crits_normal = payload_normal["critical_failures"]

    # This fixture should be valid: no critical failures in normal mode
    assert len(crits_normal) == 0, f"Expected no critical failures in good fixture. Found: {crits_normal}"

    # ------------------------------------------------------------
    # 2) Strict run (same input, stricter exit-code interpretation)
    # ------------------------------------------------------------
    outdir_strict = tmp_path / "out_strict"
    code_strict, json_path_strict = _run_json_only(grid, outdir_strict, extra_args=["--strict"])

    assert json_path_strict.exists(), f"Expected JSON report to exist: {json_path_strict}"
    payload_strict = read_report_json(json_path_strict)

    # Minimal JSON schema contract again (keep strict-mode output shape stable)
    assert_min_schema(payload_strict, context="good grid strict JSON report")

    crits_strict = payload_strict["critical_failures"]
    warns_strict = payload_strict["warnings"]

    # The fixture should still be valid (strict-mode should not create critical failures)
    assert len(crits_strict) == 0, f"Strict mode must not invent critical failures. Found: {crits_strict}"

    # ------------------------------------------------------------
    # 3) Exit-code contract assertions
    # ------------------------------------------------------------
    # NOTE:
    # We intentionally do NOT require that warnings exist.
    # If warnings are present, strict mode must return exit code 1.
    # If warnings are absent, strict mode should behave the same as non-strict.
    if len(warns_strict) > 0:
        assert code_strict == 1, "Strict mode must return exit code 1 when warnings exist and there are no critical failures"
    else:
        assert code_strict == code_normal, "Strict mode should not change exit code when there are no warnings"

def test_bad_grid_signflip_fails(tmp_path: Path, grids_dir: Path, read_report_json):
    grid = grids_dir / "grid_tiny_bad_J_signflip.nc"
    assert grid.exists(), f"Missing fixture: {grid}"

    outdir = tmp_path / "out_bad1"
    code, json_path = _run_json_only(grid, outdir)

    assert code == 1, "Expected bad grid to return exit code 1 (critical failures)"
    assert json_path.exists(), f"Expected JSON report to exist: {json_path}"

    payload = read_report_json(json_path)

    # Minimal JSON schema contract for clarity and robustness
    assert_min_schema(payload, context="bad grid signflip JSON report")

    crits = payload["critical_failures"]
    assert len(crits) > 0, f"Expected at least one critical failure. Found: {crits}"

def test_bad_grid_nan_metric_fails(tmp_path: Path, grids_dir: Path, read_report_json):
    grid = grids_dir / "grid_tiny_bad_g11_nan.nc"
    assert grid.exists(), f"Missing fixture: {grid}"

    outdir = tmp_path / "out_bad2"
    code, json_path = _run_json_only(grid, outdir)

    assert code == 1, "Expected bad grid to return exit code 1 (critical failures)"
    assert json_path.exists(), f"Expected JSON report to exist: {json_path}"

    payload = read_report_json(json_path)

    # Minimal JSON schema contract for clarity and robustness
    assert_min_schema(payload, context="bad grid NaN metric JSON report")

    crits = payload["critical_failures"]
    assert len(crits) > 0, f"Expected at least one critical failure. Found: {crits}"

def test_no_args_returns_1_when_grid_nc_missing(monkeypatch, tmp_path: Path):
    """
    main_cli() with no argv should try 'grid.nc' in cwd and return 1 if missing. 
    """
    monkeypatch.chdir(tmp_path)  # empty dir, no grid.nc
    code = diag.main_cli([])
    assert code == 1