"""Tests for copper_oxide_dft.structure_builder."""

from __future__ import annotations

import numpy as np

from copper_oxide_dft.structure_builder import (
    CU_LATTICE_PARAMETER_ANG,
    build_bulk_cu,
)


def test_build_bulk_cu_default_lattice() -> None:
    atoms = build_bulk_cu()
    assert atoms.get_chemical_formula() == "Cu"
    assert len(atoms) == 1
    # fcc primitive cell volume = a^3 / 4
    expected_volume = CU_LATTICE_PARAMETER_ANG**3 / 4.0
    np.testing.assert_allclose(atoms.get_volume(), expected_volume, rtol=1e-6)


def test_build_bulk_cu_custom_lattice_parameter() -> None:
    a = 3.70
    atoms = build_bulk_cu(a=a)
    np.testing.assert_allclose(atoms.get_volume(), a**3 / 4.0, rtol=1e-6)
