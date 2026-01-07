from __future__ import annotations

from pathlib import Path
import json
import pytest

# Shared pytest fixtures for bout_tokamak_grid_diagnostics.
# Kept intentionally minimal: path resolution + JSON loader helpers only.

@pytest.fixture
def grids_dir() -> Path:
    # .../bout_tokamak_grid_diagnostics/tests/data/grids
    return Path(__file__).resolve().parent / "data" / "grids"


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def read_report_json():
    return read_json