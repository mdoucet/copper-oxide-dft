"""Tests for copper_oxide_dft.config (converged-parameter store)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from copper_oxide_dft.config import (
    SCHEMA_VERSION,
    ProjectConfig,
    SystemConfig,
    load_config,
    save_config,
)


def test_system_config_round_trip_preserves_known_fields() -> None:
    sc = SystemConfig(
        ecutwfc_ry=80.0, kpts=(8, 8, 8), degauss_ry=0.02, ecutrho_ry=640.0
    )
    restored = SystemConfig.from_dict(sc.to_dict())
    assert restored == sc


def test_system_config_preserves_unknown_extras() -> None:
    """Future phases will add keys; we must not drop them on load."""
    payload = {
        "ecutwfc_ry": 80.0,
        "kpts": [6, 6, 6],
        "degauss_ry": 0.02,
        "hubbard_u_ev": 4.0,
        "vacuum_ang": 15.0,
    }
    sc = SystemConfig.from_dict(payload)
    assert sc.extras == {"hubbard_u_ev": 4.0, "vacuum_ang": 15.0}
    # Round-trip restores all the original keys.
    assert sc.to_dict() == payload


def test_system_config_rejects_short_kpts() -> None:
    with pytest.raises(ValueError, match="kpts must be 3"):
        SystemConfig.from_dict(
            {"ecutwfc_ry": 80.0, "kpts": [8, 8], "degauss_ry": 0.02}
        )


def test_project_config_round_trip(tmp_path: Path) -> None:
    config = ProjectConfig(
        systems={
            "bulk_cu": SystemConfig(
                ecutwfc_ry=80.0, kpts=(8, 8, 8), degauss_ry=0.02
            ),
            "bulk_cu2o": SystemConfig(
                ecutwfc_ry=80.0,
                kpts=(6, 6, 6),
                degauss_ry=0.02,
                extras={"hubbard_u_ev": 4.0},
            ),
        }
    )
    p = save_config(config, tmp_path / "converged.json")
    loaded = load_config(p)
    assert set(loaded.systems.keys()) == {"bulk_cu", "bulk_cu2o"}
    assert loaded.systems["bulk_cu"].kpts == (8, 8, 8)
    assert loaded.systems["bulk_cu2o"].extras == {"hubbard_u_ev": 4.0}


def test_save_config_writes_schema_version(tmp_path: Path) -> None:
    config = ProjectConfig()
    p = save_config(config, tmp_path / "empty.json")
    payload = json.loads(p.read_text())
    assert payload["schema_version"] == SCHEMA_VERSION


def test_load_config_rejects_unknown_schema_version(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"schema_version": 999, "systems": {}}))
    with pytest.raises(ValueError, match="schema_version"):
        load_config(p)


def test_load_config_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_config(tmp_path / "nope.json")
