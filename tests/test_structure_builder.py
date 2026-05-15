"""Tests for copper_oxide_dft.structure_builder."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from copper_oxide_dft.structure_builder import (
    CU2O_LATTICE_PARAMETER_ANG,
    CU_LATTICE_PARAMETER_ANG,
    CUO_BETA_DEG,
    CUO_LATTICE_PARAMETERS_ANG,
    DEFAULT_SLAB_LAYERS,
    Layer,
    add_explicit_water_layer,
    add_oxygen_adsorbates,
    build_bulk_cu,
    build_bulk_cu2o,
    build_bulk_cuo,
    build_cu2o_111_slab,
    build_cu111_slab,
    build_cuo_111_slab,
    build_reference_h2,
    build_reference_h2o,
    summarize_layers,
    surface_energy_ev_per_a2,
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


# ---- bulk oxides -----------------------------------------------------------


def test_build_bulk_cu2o_has_4Cu_2O_in_cubic_cell() -> None:
    atoms = build_bulk_cu2o()
    symbols = atoms.get_chemical_symbols()
    assert symbols.count("Cu") == 4
    assert symbols.count("O") == 2
    # Cubic cell with default lattice parameter.
    cell = atoms.get_cell()
    np.testing.assert_allclose(cell.lengths(), [CU2O_LATTICE_PARAMETER_ANG] * 3)
    np.testing.assert_allclose(cell.angles(), [90.0, 90.0, 90.0])


def test_build_bulk_cu2o_custom_lattice_parameter_scales_volume() -> None:
    a = 4.0
    atoms = build_bulk_cu2o(a=a)
    np.testing.assert_allclose(atoms.get_volume(), a**3, rtol=1e-10)


def test_build_bulk_cuo_has_4Cu_4O_and_afm_starting_moments() -> None:
    atoms = build_bulk_cuo()
    symbols = atoms.get_chemical_symbols()
    assert symbols.count("Cu") == 4
    assert symbols.count("O") == 4
    moments = atoms.get_initial_magnetic_moments()
    cu_moments = [moments[i] for i, s in enumerate(symbols) if s == "Cu"]
    o_moments = [moments[i] for i, s in enumerate(symbols) if s == "O"]
    # AFM: Cu moments must be nonzero, sum to zero (two up, two down).
    assert all(abs(m) == 1.0 for m in cu_moments)
    assert sum(cu_moments) == 0.0
    # O moments default to zero — magnetism lives on Cu d shell.
    assert all(m == 0.0 for m in o_moments)


def test_build_bulk_cuo_monoclinic_angle_matches_default() -> None:
    atoms = build_bulk_cuo()
    angles = atoms.get_cell().angles()
    # alpha and gamma stay at 90; beta is the monoclinic angle.
    np.testing.assert_allclose(angles[0], 90.0, atol=1e-6)
    np.testing.assert_allclose(angles[2], 90.0, atol=1e-6)
    np.testing.assert_allclose(angles[1], CUO_BETA_DEG, atol=1e-4)


def test_build_bulk_cuo_custom_parameters_match_cell_lengths() -> None:
    atoms = build_bulk_cuo(a=4.5, b=3.2, c=5.0, beta_deg=95.0)
    lengths = atoms.get_cell().lengths()
    np.testing.assert_allclose(lengths, [4.5, 3.2, 5.0], atol=1e-6)
    np.testing.assert_allclose(atoms.get_cell().angles()[1], 95.0, atol=1e-4)


def test_cuo_default_lattice_parameters_match_experiment() -> None:
    # Regression: experimental tenorite values (Asbrink & Norrby 1970).
    assert CUO_LATTICE_PARAMETERS_ANG[0] == pytest.approx(4.6837)
    assert CUO_LATTICE_PARAMETERS_ANG[1] == pytest.approx(3.4226)
    assert CUO_LATTICE_PARAMETERS_ANG[2] == pytest.approx(5.1288)


# ---- reference molecules ---------------------------------------------------


def test_build_reference_h2_two_atoms_correct_bond_length() -> None:
    h2 = build_reference_h2(box_size_ang=10.0)
    assert h2.get_chemical_symbols() == ["H", "H"]
    distance = float(np.linalg.norm(h2.positions[1] - h2.positions[0]))
    np.testing.assert_allclose(distance, 0.7414, atol=1e-4)
    np.testing.assert_allclose(h2.get_cell().lengths(), [10.0, 10.0, 10.0])


def test_build_reference_h2o_geometry_within_tolerances() -> None:
    h2o = build_reference_h2o(box_size_ang=12.0)
    symbols = h2o.get_chemical_symbols()
    assert symbols == ["O", "H", "H"]
    oh_distances = [
        float(np.linalg.norm(h2o.positions[i] - h2o.positions[0])) for i in (1, 2)
    ]
    np.testing.assert_allclose(oh_distances, [0.9572, 0.9572], atol=1e-4)
    # H-O-H angle from the two OH vectors.
    v1 = h2o.positions[1] - h2o.positions[0]
    v2 = h2o.positions[2] - h2o.positions[0]
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle_deg = float(np.degrees(np.arccos(cos_theta)))
    np.testing.assert_allclose(angle_deg, 104.5, atol=0.1)


def test_build_reference_h2_box_size_controls_cell() -> None:
    h2 = build_reference_h2(box_size_ang=8.0)
    np.testing.assert_allclose(h2.get_cell().lengths(), [8.0, 8.0, 8.0])


# ---- Phase 3: slab builder + adsorbates + surface energy -------------------


def test_build_cu111_slab_default_geometry_has_expected_atom_count() -> None:
    # 4 layers x 3x3 lateral = 36 atoms.
    slab = build_cu111_slab(layers=4, supercell=(3, 3))
    assert slab.get_chemical_formula() == "Cu36"
    assert len(slab) == 4 * 3 * 3


def test_build_cu111_slab_constraint_fixes_bottom_two_layers() -> None:
    slab = build_cu111_slab(layers=4, supercell=(2, 2))
    # bottom 2 layers => 2x2 lateral * 2 layers = 8 fixed atoms for 2x2 cell.
    fixed_indices = sum((list(c.index) for c in slab.constraints), [])
    assert len(fixed_indices) == 2 * 2 * 2


def test_build_cu111_slab_vacuum_increases_z_extent() -> None:
    """ASE's fcc111 adds the vacuum argument symmetrically (both sides), so
    doubling the requested vacuum adds 2x to the z cell."""
    a = build_cu111_slab(layers=4, vacuum_ang=10.0)
    b = build_cu111_slab(layers=4, vacuum_ang=20.0)
    dz = float(b.cell[2, 2]) - float(a.cell[2, 2])
    assert dz == pytest.approx(20.0, abs=0.5)


def test_build_cu111_slab_rejects_overconstrained_layers() -> None:
    with pytest.raises(ValueError, match="Cannot fix"):
        build_cu111_slab(layers=2, fix_bottom_layers=5)


def test_default_slab_layers_matches_plan() -> None:
    # The implementation plan picks 4 layers as the project standard.
    assert DEFAULT_SLAB_LAYERS == 4


def test_add_oxygen_adsorbates_one_ninth_ml_on_3x3_adds_one_o() -> None:
    slab = build_cu111_slab(supercell=(3, 3))
    covered = add_oxygen_adsorbates(slab, coverage_ml=1.0 / 9.0)
    symbols = covered.get_chemical_symbols()
    assert symbols.count("O") == 1
    assert symbols.count("Cu") == 36  # original Cu count unchanged


def test_add_oxygen_adsorbates_full_ml_on_2x2_adds_four() -> None:
    slab = build_cu111_slab(supercell=(2, 2))
    covered = add_oxygen_adsorbates(slab, coverage_ml=1.0, site="fcc")
    assert covered.get_chemical_symbols().count("O") == 4


def test_add_oxygen_adsorbates_oh_adds_one_o_and_one_h() -> None:
    slab = build_cu111_slab(supercell=(3, 3))
    covered = add_oxygen_adsorbates(slab, coverage_ml=1.0 / 9.0, adsorbate="OH")
    symbols = covered.get_chemical_symbols()
    assert symbols.count("O") == 1
    assert symbols.count("H") == 1


def test_add_oxygen_adsorbates_rejects_zero_rounded_coverage() -> None:
    slab = build_cu111_slab(supercell=(2, 2))
    with pytest.raises(ValueError, match="rounds to zero"):
        add_oxygen_adsorbates(slab, coverage_ml=0.01)  # 0.01 * 4 ≈ 0


def test_add_oxygen_adsorbates_rejects_out_of_range_coverage() -> None:
    slab = build_cu111_slab(supercell=(2, 2))
    for bad in (-0.1, 0.0, 1.5):
        with pytest.raises(ValueError, match="coverage_ml"):
            add_oxygen_adsorbates(slab, coverage_ml=bad)


def test_add_oxygen_adsorbates_rejects_unknown_site() -> None:
    slab = build_cu111_slab(supercell=(2, 2))
    with pytest.raises(ValueError, match="Unknown site"):
        add_oxygen_adsorbates(slab, coverage_ml=0.25, site="nonsense")


def test_surface_energy_helper_recovers_zero_for_bulk_match() -> None:
    """If the slab is exactly N atoms of bulk Cu, the cleave energy is 0."""
    gamma = surface_energy_ev_per_a2(
        slab_energy_ev=36 * -100.0,
        bulk_energy_per_atom_ev=-100.0,
        n_atoms_in_slab=36,
        surface_area_ang2=50.0,
        n_surfaces=2,
    )
    assert gamma == 0.0


def test_surface_energy_helper_handles_n_surfaces_one() -> None:
    """Dipole-corrected asymmetric slab has only one exposed surface."""
    gamma = surface_energy_ev_per_a2(
        slab_energy_ev=-3590.0,
        bulk_energy_per_atom_ev=-100.0,
        n_atoms_in_slab=36,
        surface_area_ang2=50.0,
        n_surfaces=1,
    )
    # (slab - N*E_bulk) = -3590 - (-3600) = +10 eV  ;  / (1 * 50) = 0.2 eV/Å^2
    assert gamma == pytest.approx(0.2)


def test_surface_energy_helper_rejects_zero_area() -> None:
    with pytest.raises(ValueError, match="surface_area_ang2 must be positive"):
        surface_energy_ev_per_a2(
            slab_energy_ev=0, bulk_energy_per_atom_ev=0,
            n_atoms_in_slab=1, surface_area_ang2=0,
        )


def test_build_cu2o_111_slab_contains_cu_and_o() -> None:
    slab = build_cu2o_111_slab(layers=2)
    syms = slab.get_chemical_symbols()
    assert "Cu" in syms
    assert "O" in syms


def test_build_cuo_111_slab_contains_cu_and_o() -> None:
    slab = build_cuo_111_slab(layers=2)
    syms = slab.get_chemical_symbols()
    assert "Cu" in syms
    assert "O" in syms


# ---- explicit water layer (Phase 6 prep) -----------------------------------


def test_add_explicit_water_layer_adds_three_atoms_per_water() -> None:
    slab = build_cu111_slab(supercell=(3, 3))
    n_slab = len(slab)
    with_water = add_explicit_water_layer(slab, n_waters=4)
    assert len(with_water) == n_slab + 4 * 3
    syms = with_water.get_chemical_symbols()
    assert syms.count("O") == 4
    assert syms.count("H") == 8


def test_add_explicit_water_layer_zero_waters_returns_copy() -> None:
    slab = build_cu111_slab(supercell=(2, 2))
    result = add_explicit_water_layer(slab, n_waters=0)
    assert len(result) == len(slab)
    # Must be a copy, not the same object.
    assert result is not slab


def test_add_explicit_water_layer_places_above_slab() -> None:
    slab = build_cu111_slab(supercell=(2, 2))
    top_z = max(atom.z for atom in slab)
    result = add_explicit_water_layer(slab, n_waters=2, height_ang=3.0)
    water_indices = list(range(len(slab), len(result)))
    water_z = [result[i].z for i in water_indices]
    # All water atoms sit above the topmost slab atom plus the requested height.
    assert min(water_z) >= top_z + 3.0 - 0.1  # tolerance for H offset


def test_add_explicit_water_layer_rejects_negative_count() -> None:
    slab = build_cu111_slab(supercell=(2, 2))
    with pytest.raises(ValueError, match="non-negative"):
        add_explicit_water_layer(slab, n_waters=-1)


def test_add_explicit_water_layer_deterministic_seed() -> None:
    """Same seed → identical placements (regression for reproducibility)."""
    slab = build_cu111_slab(supercell=(3, 3))
    a = add_explicit_water_layer(slab, n_waters=3, seed=42)
    b = add_explicit_water_layer(slab, n_waters=3, seed=42)
    np.testing.assert_allclose(a.positions, b.positions)
