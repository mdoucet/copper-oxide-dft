"""Tests for copper_oxide_dft.structure_builder."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from copper_oxide_dft.structure_builder import (
    CU_LATTICE_PARAMETER_ANG,
    Layer,
    build_bulk_cu,
    summarize_layers,
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


# ---- summarize_layers ------------------------------------------------------


def _slab(positions: list[tuple[float, float, float]], symbols: str) -> Atoms:
    return Atoms(symbols=symbols, positions=positions, cell=[10, 10, 30], pbc=True)


def test_summarize_layers_bulk_cu_is_single_layer() -> None:
    atoms = build_bulk_cu()
    layers = summarize_layers(atoms)
    assert len(layers) == 1
    assert layers[0].total_atoms == 1
    assert layers[0].elements == {"Cu": 1}


def test_summarize_layers_empty_structure_returns_empty_list() -> None:
    assert summarize_layers(Atoms()) == []


def test_summarize_layers_groups_within_tolerance() -> None:
    # Three planes at z=0, z=2.1, z=4.2 with one Cu atom each; the small
    # jitter (0.05) should be absorbed into a single layer.
    atoms = _slab(
        [(0, 0, 0.00), (1, 0, 0.05), (0, 1, 2.10), (1, 1, 2.08), (2, 2, 4.20)],
        "Cu5",
    )
    layers = summarize_layers(atoms, tol=0.1)
    assert [layer.total_atoms for layer in layers] == [2, 2, 1]
    # Z-coordinates returned in ascending order.
    z_values = [layer.z for layer in layers]
    assert z_values == sorted(z_values)


def test_summarize_layers_orders_by_z() -> None:
    # Add atoms out of order; result should still be bottom-to-top.
    atoms = _slab([(0, 0, 5.0), (0, 0, 0.0), (0, 0, 2.5)], "Cu3")
    layers = summarize_layers(atoms)
    z_values = [layer.z for layer in layers]
    assert z_values == [0.0, 2.5, 5.0]


def test_summarize_layers_mixed_species_in_layer() -> None:
    # Surface oxide: a Cu layer and a half-coverage O layer at slightly
    # higher z.
    atoms = _slab(
        [(0, 0, 0.0), (1, 0, 0.0), (0, 1, 0.0), (0, 0, 1.5), (1, 0, 1.5)],
        "Cu3O2",
    )
    layers = summarize_layers(atoms, tol=0.1)
    assert layers[0].elements == {"Cu": 3}
    assert layers[1].elements == {"O": 2}


def test_summarize_layers_rejects_negative_tolerance() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        summarize_layers(build_bulk_cu(), tol=-0.1)


def test_layer_composition_label_sorted_alphabetically() -> None:
    layer = Layer(z=0.0, elements={"O": 2, "Cu": 3}, thickness=0.0)
    assert layer.composition_label() == "Cux3 Ox2"
