"""Tests for copper_oxide_dft.convergence."""

from __future__ import annotations

from pathlib import Path

import pytest

from copper_oxide_dft.convergence import (
    SUPPORTED_SWEEP_PARAMETERS,
    sweep_convergence,
)
from copper_oxide_dft.structure_builder import build_bulk_cu, build_bulk_cu2o


@pytest.fixture
def pseudo_dir(tmp_path: Path) -> Path:
    d = tmp_path / "pseudos"
    d.mkdir()
    (d / "Cu.upf").write_text("")
    return d


def test_sweep_ecutwfc_creates_one_input_per_value(
    tmp_path: Path, pseudo_dir: Path
) -> None:
    atoms = build_bulk_cu()
    values = [40.0, 60.0, 80.0]
    paths = sweep_convergence(
        atoms,
        out_root=tmp_path / "conv",
        pseudopotentials={"Cu": "Cu.upf"},
        param="ecutwfc",
        values=values,
        pseudo_dir=pseudo_dir,
    )

    assert len(paths) == len(values)
    for value, path in zip(values, paths, strict=True):
        assert path.is_file()
        assert path.name == "pw.in"
        assert path.parent.name == f"ecutwfc_{value:.0f}"
        text = path.read_text()
        assert (
            f"ecutwfc          = {value}" in text
            or f"ecutwfc          ={value:>9}" in text
        )


def test_sweep_kpts_expands_int_to_3tuple(tmp_path: Path, pseudo_dir: Path) -> None:
    atoms = build_bulk_cu()
    paths = sweep_convergence(
        atoms,
        out_root=tmp_path / "conv",
        pseudopotentials={"Cu": "Cu.upf"},
        param="kpts",
        values=[6, 8, 10],
        pseudo_dir=pseudo_dir,
    )

    assert {p.parent.name for p in paths} == {"kpts_6", "kpts_8", "kpts_10"}
    text = (tmp_path / "conv" / "kpts_8" / "pw.in").read_text()
    assert "8 8 8" in text


def test_sweep_degauss_uses_p_in_label(tmp_path: Path, pseudo_dir: Path) -> None:
    atoms = build_bulk_cu()
    paths = sweep_convergence(
        atoms,
        out_root=tmp_path / "conv",
        pseudopotentials={"Cu": "Cu.upf"},
        param="degauss",
        values=[0.01, 0.02, 0.03],
        pseudo_dir=pseudo_dir,
    )

    # Period replaced with "p" so directory names stay filesystem-friendly.
    assert {p.parent.name for p in paths} == {
        "degauss_0p010",
        "degauss_0p020",
        "degauss_0p030",
    }


def test_sweep_base_kwargs_threaded_into_each_input(
    tmp_path: Path, pseudo_dir: Path
) -> None:
    atoms = build_bulk_cu()
    paths = sweep_convergence(
        atoms,
        out_root=tmp_path / "conv",
        pseudopotentials={"Cu": "Cu.upf"},
        param="kpts",
        values=[4, 6],
        pseudo_dir=pseudo_dir,
        base_kwargs={"ecutwfc": 50.0, "degauss": 0.015},
    )
    for path in paths:
        text = path.read_text()
        assert "50.0" in text  # ecutwfc held constant during kpts sweep
        assert "0.015" in text  # degauss held constant


def test_sweep_rejects_unknown_param(tmp_path: Path, pseudo_dir: Path) -> None:
    atoms = build_bulk_cu()
    with pytest.raises(ValueError, match="Unsupported sweep param"):
        sweep_convergence(
            atoms,
            out_root=tmp_path / "conv",
            pseudopotentials={"Cu": "Cu.upf"},
            param="mixing_beta",
            values=[0.3, 0.5],
            pseudo_dir=pseudo_dir,
        )


def test_sweep_rejects_param_in_base_kwargs(tmp_path: Path, pseudo_dir: Path) -> None:
    atoms = build_bulk_cu()
    with pytest.raises(ValueError, match="also appears in base_kwargs"):
        sweep_convergence(
            atoms,
            out_root=tmp_path / "conv",
            pseudopotentials={"Cu": "Cu.upf"},
            param="ecutwfc",
            values=[40.0, 60.0],
            pseudo_dir=pseudo_dir,
            base_kwargs={"ecutwfc": 80.0},
        )


def test_supported_sweep_parameters_set() -> None:
    assert frozenset(
        {"ecutwfc", "kpts", "degauss", "hubbard_u"}
    ) == SUPPORTED_SWEEP_PARAMETERS


def test_sweep_hubbard_u_writes_one_input_per_value(
    tmp_path: Path, pseudo_dir: Path
) -> None:
    """U sweep on Cu2O: each input must carry Hubbard_U(1) with the swept value."""
    (pseudo_dir / "O.upf").write_text("")
    atoms = build_bulk_cu2o()
    paths = sweep_convergence(
        atoms,
        out_root=tmp_path / "u_sweep",
        pseudopotentials={"Cu": "Cu.upf", "O": "O.upf"},
        param="hubbard_u",
        values=[0.0, 2.0, 4.0, 6.0],
        pseudo_dir=pseudo_dir,
    )
    assert {p.parent.name for p in paths} == {
        "hubbard_u_0p00",
        "hubbard_u_2p00",
        "hubbard_u_4p00",
        "hubbard_u_6p00",
    }
    text = (tmp_path / "u_sweep" / "hubbard_u_4p00" / "pw.in").read_text().lower()
    # Cu is species 1 in Cu2O; non-magnetic so nspin defaults to 1.
    assert "hubbard_u(1)" in text
    assert "4.0" in text
    assert "nspin" in text
