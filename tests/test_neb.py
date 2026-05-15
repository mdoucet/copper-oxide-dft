"""Tests for copper_oxide_dft.neb (Phase 8 NEB input writer)."""

from __future__ import annotations

from pathlib import Path

import pytest

from copper_oxide_dft.neb import write_neb_input
from copper_oxide_dft.structure_builder import (
    add_oxygen_adsorbates,
    build_cu111_slab,
)


@pytest.fixture
def pseudo_dir(tmp_path: Path) -> Path:
    d = tmp_path / "pseudos"
    d.mkdir()
    for name in ("Cu.upf", "O.upf"):
        (d / name).write_text("")
    return d


def _endpoints():
    slab = build_cu111_slab(supercell=(2, 2))
    initial = add_oxygen_adsorbates(slab, coverage_ml=0.25, site="fcc")
    final = add_oxygen_adsorbates(slab, coverage_ml=0.25, site="hcp")
    return initial, final


def test_write_neb_input_emits_path_namelist(tmp_path: Path, pseudo_dir: Path) -> None:
    initial, final = _endpoints()
    written = write_neb_input(
        tmp_path / "neb.in",
        endpoints=(initial, final),
        n_intermediate_images=3,
        pseudopotentials={"Cu": "Cu.upf", "O": "O.upf"},
        pseudo_dir=pseudo_dir,
    )
    text = written.read_text()
    assert "BEGIN_PATH_INPUT" in text
    assert "&PATH" in text
    assert "string_method = 'neb'" in text
    assert "num_of_images = 5" in text  # 3 intermediate + 2 endpoints
    assert "FIRST_IMAGE" in text
    assert "LAST_IMAGE" in text
    assert "ATOMIC_POSITIONS angstrom" in text
    assert "END_ENGINE_INPUT" in text


def test_write_neb_input_rejects_too_few_images(tmp_path: Path, pseudo_dir: Path) -> None:
    initial, final = _endpoints()
    with pytest.raises(ValueError, match="n_intermediate_images"):
        write_neb_input(
            tmp_path / "neb.in",
            endpoints=(initial, final),
            n_intermediate_images=0,
            pseudopotentials={"Cu": "Cu.upf", "O": "O.upf"},
            pseudo_dir=pseudo_dir,
        )


def test_write_neb_input_rejects_mismatched_endpoints(
    tmp_path: Path, pseudo_dir: Path
) -> None:
    initial, _ = _endpoints()
    # Add an extra atom to final so chemical formulas diverge.
    final = initial.copy()
    final.append(initial[0])  # duplicate first atom
    with pytest.raises(ValueError, match="atom counts differ|formulas"):
        write_neb_input(
            tmp_path / "neb.in",
            endpoints=(initial, final),
            n_intermediate_images=3,
            pseudopotentials={"Cu": "Cu.upf", "O": "O.upf"},
            pseudo_dir=pseudo_dir,
        )


def test_write_neb_input_ci_scheme_propagates(
    tmp_path: Path, pseudo_dir: Path
) -> None:
    initial, final = _endpoints()
    written = write_neb_input(
        tmp_path / "neb.in",
        endpoints=(initial, final),
        n_intermediate_images=2,
        pseudopotentials={"Cu": "Cu.upf", "O": "O.upf"},
        pseudo_dir=pseudo_dir,
        ci_scheme="no-CI",
    )
    assert "CI_scheme = 'no-CI'" in written.read_text()


def test_write_neb_input_extra_path_overrides_defaults(
    tmp_path: Path, pseudo_dir: Path
) -> None:
    initial, final = _endpoints()
    written = write_neb_input(
        tmp_path / "neb.in",
        endpoints=(initial, final),
        n_intermediate_images=2,
        pseudopotentials={"Cu": "Cu.upf", "O": "O.upf"},
        pseudo_dir=pseudo_dir,
        extra_path={"first_last_opt": True},
    )
    assert "first_last_opt = .true." in written.read_text()
