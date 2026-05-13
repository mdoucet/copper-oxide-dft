"""Tests for copper_oxide_dft.qe_input."""

from __future__ import annotations

from pathlib import Path

import pytest

from copper_oxide_dft.qe_input import PSEUDO_DIR_ENV_VAR, write_scf_input
from copper_oxide_dft.structure_builder import build_bulk_cu


@pytest.fixture
def pseudo_dir(tmp_path: Path) -> Path:
    d = tmp_path / "pseudos"
    d.mkdir()
    (d / "Cu.upf").write_text("")
    return d


def test_write_scf_input_explicit_pseudo_dir(tmp_path: Path, pseudo_dir: Path) -> None:
    atoms = build_bulk_cu()
    out_path = tmp_path / "run" / "bulk_cu.in"

    written = write_scf_input(
        atoms,
        out_path=out_path,
        pseudopotentials={"Cu": "Cu.upf"},
        ecutwfc=60.0,
        kpts=(6, 6, 6),
        pseudo_dir=pseudo_dir,
    )

    assert written == out_path
    text = out_path.read_text()
    # Namelists
    assert "&CONTROL" in text
    assert "&SYSTEM" in text
    assert "&ELECTRONS" in text
    # Project-standard parameters
    assert "calculation" in text and "scf" in text
    assert "ecutwfc" in text and "60" in text
    assert "smearing" in text and "mv" in text.lower()  # Marzari-Vanderbilt
    assert "K_POINTS" in text
    assert "Cu.upf" in text


def test_write_scf_input_falls_back_to_env_var(
    tmp_path: Path, pseudo_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(PSEUDO_DIR_ENV_VAR, str(pseudo_dir))
    atoms = build_bulk_cu()
    out_path = tmp_path / "bulk_cu.in"

    write_scf_input(
        atoms,
        out_path=out_path,
        pseudopotentials={"Cu": "Cu.upf"},
    )

    assert out_path.is_file()


def test_write_scf_input_raises_when_no_pseudo_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(PSEUDO_DIR_ENV_VAR, raising=False)
    atoms = build_bulk_cu()
    with pytest.raises(FileNotFoundError, match=PSEUDO_DIR_ENV_VAR):
        write_scf_input(
            atoms,
            out_path=tmp_path / "x.in",
            pseudopotentials={"Cu": "Cu.upf"},
        )


def test_write_scf_input_raises_when_pseudo_dir_missing(tmp_path: Path) -> None:
    atoms = build_bulk_cu()
    with pytest.raises(FileNotFoundError, match="does not exist"):
        write_scf_input(
            atoms,
            out_path=tmp_path / "x.in",
            pseudopotentials={"Cu": "Cu.upf"},
            pseudo_dir=tmp_path / "nonexistent",
        )


def test_write_scf_input_extra_namelist_override(
    tmp_path: Path, pseudo_dir: Path
) -> None:
    """Spin-polarized override is the use case we'll need for CuO later."""
    atoms = build_bulk_cu()
    out_path = tmp_path / "bulk_cu.in"
    write_scf_input(
        atoms,
        out_path=out_path,
        pseudopotentials={"Cu": "Cu.upf"},
        pseudo_dir=pseudo_dir,
        extra_input_data={"system": {"nspin": 2}},
    )
    assert "nspin" in out_path.read_text()
