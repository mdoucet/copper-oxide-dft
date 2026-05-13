"""Tests for copper_oxide_dft.qe_input."""

from __future__ import annotations

from pathlib import Path

import pytest

from copper_oxide_dft.qe_input import (
    PSEUDO_DIR_ENV_VAR,
    SUPPORTED_CALCULATIONS,
    write_pw_input,
)
from copper_oxide_dft.structure_builder import build_bulk_cu


@pytest.fixture
def pseudo_dir(tmp_path: Path) -> Path:
    d = tmp_path / "pseudos"
    d.mkdir()
    (d / "Cu.upf").write_text("")
    return d


def test_write_pw_input_explicit_pseudo_dir(tmp_path: Path, pseudo_dir: Path) -> None:
    atoms = build_bulk_cu()
    out_path = tmp_path / "run" / "bulk_cu.in"

    written = write_pw_input(
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


def test_write_pw_input_falls_back_to_env_var(
    tmp_path: Path, pseudo_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(PSEUDO_DIR_ENV_VAR, str(pseudo_dir))
    atoms = build_bulk_cu()
    out_path = tmp_path / "bulk_cu.in"

    write_pw_input(
        atoms,
        out_path=out_path,
        pseudopotentials={"Cu": "Cu.upf"},
    )

    assert out_path.is_file()


def test_write_pw_input_raises_when_no_pseudo_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(PSEUDO_DIR_ENV_VAR, raising=False)
    atoms = build_bulk_cu()
    with pytest.raises(FileNotFoundError, match=PSEUDO_DIR_ENV_VAR):
        write_pw_input(
            atoms,
            out_path=tmp_path / "x.in",
            pseudopotentials={"Cu": "Cu.upf"},
        )


def test_write_pw_input_raises_when_pseudo_dir_missing(tmp_path: Path) -> None:
    atoms = build_bulk_cu()
    with pytest.raises(FileNotFoundError, match="does not exist"):
        write_pw_input(
            atoms,
            out_path=tmp_path / "x.in",
            pseudopotentials={"Cu": "Cu.upf"},
            pseudo_dir=tmp_path / "nonexistent",
        )


def test_write_pw_input_extra_namelist_override(
    tmp_path: Path, pseudo_dir: Path
) -> None:
    """Spin-polarized override is the use case we'll need for CuO later."""
    atoms = build_bulk_cu()
    out_path = tmp_path / "bulk_cu.in"
    write_pw_input(
        atoms,
        out_path=out_path,
        pseudopotentials={"Cu": "Cu.upf"},
        pseudo_dir=pseudo_dir,
        extra_input_data={"system": {"nspin": 2}},
    )
    assert "nspin" in out_path.read_text()


def test_write_pw_input_vc_relax_adds_ions_and_cell_namelists(
    tmp_path: Path, pseudo_dir: Path
) -> None:
    """vc-relax must emit IONS + CELL namelists with BFGS dynamics; this is
    what we'll use for Phase 1 bulk-Cu lattice optimization."""
    atoms = build_bulk_cu()
    out_path = tmp_path / "vc_relax.in"
    write_pw_input(
        atoms,
        out_path=out_path,
        pseudopotentials={"Cu": "Cu.upf"},
        pseudo_dir=pseudo_dir,
        calculation="vc-relax",
    )
    text = out_path.read_text()
    assert "'vc-relax'" in text
    assert "ion_dynamics" in text
    assert "cell_dynamics" in text
    assert "bfgs" in text


def test_write_pw_input_relax_adds_ions_but_not_cell(
    tmp_path: Path, pseudo_dir: Path
) -> None:
    atoms = build_bulk_cu()
    out_path = tmp_path / "relax.in"
    write_pw_input(
        atoms,
        out_path=out_path,
        pseudopotentials={"Cu": "Cu.upf"},
        pseudo_dir=pseudo_dir,
        calculation="relax",
    )
    text = out_path.read_text()
    assert "'relax'" in text
    assert "ion_dynamics" in text
    assert "cell_dynamics" not in text


def test_write_pw_input_rejects_bad_calculation(
    tmp_path: Path, pseudo_dir: Path
) -> None:
    atoms = build_bulk_cu()
    with pytest.raises(ValueError, match="Unsupported calculation"):
        write_pw_input(
            atoms,
            out_path=tmp_path / "x.in",
            pseudopotentials={"Cu": "Cu.upf"},
            pseudo_dir=pseudo_dir,
            calculation="totally-not-a-thing",
        )


def test_supported_calculations_includes_expected_modes() -> None:
    assert {"scf", "relax", "vc-relax"}.issubset(SUPPORTED_CALCULATIONS)
