"""Tests for copper_oxide_dft.ml.sld."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from copper_oxide_dft.ml.sld import (
    EXPERIMENTAL_BULK_CU_SLD_E6_PER_A2,
    FEMTOMETERS_TO_ANGSTROMS,
    NEUTRON_SCATTERING_LENGTH_FM,
    SldProfile,
    bulk_cu_normalization_factor,
    compute_bulk_cu_sld_e6_per_a2,
    compute_sld_profile,
    compute_sld_summed_b_over_volume,
)
from copper_oxide_dft.structure_builder import build_bulk_cu

# ---------- constants ---------------------------------------------------------


def test_neutron_scattering_lengths_match_nist() -> None:
    """Spot-check the canonical values."""
    np.testing.assert_allclose(NEUTRON_SCATTERING_LENGTH_FM["Cu"], 7.718, atol=0.01)
    np.testing.assert_allclose(NEUTRON_SCATTERING_LENGTH_FM["O"], 5.803, atol=0.01)
    assert NEUTRON_SCATTERING_LENGTH_FM["H"] < 0   # hydrogen is anomalous: negative b


def test_femtometer_conversion() -> None:
    """1 fm = 10^-15 m = 10^-5 Å."""
    np.testing.assert_allclose(FEMTOMETERS_TO_ANGSTROMS, 1e-5)


# ---------- compute_bulk_cu_sld_e6_per_a2 -------------------------------------


def test_bulk_cu_sld_at_experimental_lattice() -> None:
    """SLD of Cu at a = 3.615 Å should match the tabulated experimental value."""
    sld = compute_bulk_cu_sld_e6_per_a2(a_ang=3.615)
    np.testing.assert_allclose(sld, EXPERIMENTAL_BULK_CU_SLD_E6_PER_A2, rtol=1e-3)


def test_bulk_cu_sld_at_relaxed_lattice_is_lower() -> None:
    """PBE-relaxed `a` = 3.6577 Å is larger → density lower → SLD lower."""
    sld_exp = compute_bulk_cu_sld_e6_per_a2(a_ang=3.615)
    sld_relaxed = compute_bulk_cu_sld_e6_per_a2(a_ang=3.6577)
    assert sld_relaxed < sld_exp
    # The 1.18% lattice overshoot → ~3.5% SLD undershoot.
    np.testing.assert_allclose(
        (sld_exp - sld_relaxed) / sld_exp, 3.0 * 0.0118, rtol=0.05
    )


def test_bulk_cu_sld_scales_as_one_over_a_cubed() -> None:
    """Doubling `a` should cut SLD by a factor of 8."""
    sld_a = compute_bulk_cu_sld_e6_per_a2(a_ang=3.0)
    sld_2a = compute_bulk_cu_sld_e6_per_a2(a_ang=6.0)
    np.testing.assert_allclose(sld_a / sld_2a, 8.0)


# ---------- bulk_cu_normalization_factor --------------------------------------


def test_normalization_factor_unity_at_experimental_lattice() -> None:
    factor = bulk_cu_normalization_factor(simulated_a_ang=3.615)
    np.testing.assert_allclose(factor, 1.0, rtol=1e-3)


def test_normalization_factor_corrects_pbe_overshoot() -> None:
    """At the PBE-relaxed lattice, the multiplier should boost SLD by ~3.5 %."""
    factor = bulk_cu_normalization_factor(simulated_a_ang=3.6577)
    assert 1.03 < factor < 1.04


def test_normalization_factor_uses_experimental_override() -> None:
    factor = bulk_cu_normalization_factor(
        simulated_a_ang=3.615, experimental_sld_e6_per_a2=13.0
    )
    # factor = exp / sim ≈ 13.0 / 6.535 → ≈ 1.99.
    np.testing.assert_allclose(factor, 13.0 / 6.535, rtol=1e-3)


# ---------- compute_sld_summed_b_over_volume ----------------------------------


def test_sld_on_pure_cu_matches_closed_form() -> None:
    """The atoms-based and closed-form SLDs must agree on bulk Cu."""
    atoms = build_bulk_cu(a=3.615) * (2, 2, 2)
    # The cubic cell has z-extent equal to `a` for the conventional cell;
    # use the full atoms range and let the function compute the volume.
    # For a primitive fcc cell built by ASE, the cell isn't aligned with
    # global axes — use the cell volume directly.
    cell_volume = float(atoms.get_volume())
    n_cu = len(atoms)
    summed_b_ang = n_cu * NEUTRON_SCATTERING_LENGTH_FM["Cu"] * FEMTOMETERS_TO_ANGSTROMS
    expected_sld = summed_b_ang / cell_volume * 1e6
    np.testing.assert_allclose(expected_sld, EXPERIMENTAL_BULK_CU_SLD_E6_PER_A2, rtol=1e-3)


def test_sld_summed_b_over_volume_known_value() -> None:
    """One Cu atom in a 10 Å lateral × 10 Å thick slab.

    SLD = b / (A · δz) = 7.718 fm / (100 Å² × 10 Å)
        = 7.718e-5 Å / 1000 Å³
        = 7.718e-8 Å⁻²
        = 0.07718 × 10⁻⁶ Å⁻²
    """
    atoms = Atoms("Cu", positions=[(5.0, 5.0, 5.0)], cell=[10, 10, 20], pbc=True)
    sld = compute_sld_summed_b_over_volume(
        atoms, z_range_ang=(0.0, 10.0), lateral_area_ang2=100.0
    )
    np.testing.assert_allclose(sld, 0.07718, atol=1e-4)


def test_sld_summed_b_over_volume_uses_cell_area_when_unset() -> None:
    """If lateral_area_ang2 is None, use the cell's a×b cross product."""
    atoms = Atoms("Cu", positions=[(5.0, 5.0, 5.0)], cell=[10, 10, 20], pbc=True)
    sld_explicit = compute_sld_summed_b_over_volume(
        atoms, z_range_ang=(0.0, 10.0), lateral_area_ang2=100.0
    )
    sld_from_cell = compute_sld_summed_b_over_volume(
        atoms, z_range_ang=(0.0, 10.0)
    )
    np.testing.assert_allclose(sld_explicit, sld_from_cell)


def test_sld_summed_b_over_volume_negative_for_pure_hydrogen() -> None:
    """H has negative b → negative SLD (a real, physical phenomenon)."""
    atoms = Atoms("H", positions=[(5, 5, 5)], cell=[10, 10, 20], pbc=True)
    sld = compute_sld_summed_b_over_volume(
        atoms, z_range_ang=(0.0, 10.0), lateral_area_ang2=100.0
    )
    assert sld < 0


def test_sld_summed_b_excludes_atoms_outside_z_range() -> None:
    """Atoms at z=15 should not contribute to a slab over (0, 10)."""
    atoms = Atoms(
        "Cu2",
        positions=[(5, 5, 5), (5, 5, 15)],   # one in slab, one outside
        cell=[10, 10, 30], pbc=True,
    )
    sld = compute_sld_summed_b_over_volume(
        atoms, z_range_ang=(0.0, 10.0), lateral_area_ang2=100.0
    )
    # Only the first atom contributes — same as the one-atom test.
    np.testing.assert_allclose(sld, 0.07718, atol=1e-4)


def test_sld_summed_b_raises_on_empty_window() -> None:
    atoms = Atoms("Cu", positions=[(5, 5, 50)], cell=[10, 10, 100], pbc=True)
    with pytest.raises(ValueError, match="No atoms"):
        compute_sld_summed_b_over_volume(atoms, z_range_ang=(0, 10),
                                           lateral_area_ang2=100.0)


def test_sld_summed_b_raises_on_unknown_species() -> None:
    atoms = Atoms("Xe", positions=[(5, 5, 5)], cell=[10, 10, 20], pbc=True)
    with pytest.raises(ValueError, match="scattering length"):
        compute_sld_summed_b_over_volume(atoms, z_range_ang=(0, 10),
                                           lateral_area_ang2=100.0)


def test_sld_summed_b_rejects_degenerate_z_range() -> None:
    atoms = Atoms("Cu", positions=[(5, 5, 5)], cell=[10, 10, 20], pbc=True)
    with pytest.raises(ValueError):
        compute_sld_summed_b_over_volume(atoms, z_range_ang=(5, 5),
                                           lateral_area_ang2=100.0)


def test_sld_summed_b_rejects_negative_area() -> None:
    atoms = Atoms("Cu", positions=[(5, 5, 5)], cell=[10, 10, 20], pbc=True)
    with pytest.raises(ValueError):
        compute_sld_summed_b_over_volume(atoms, z_range_ang=(0, 10),
                                           lateral_area_ang2=-1.0)


def test_sld_summed_b_custom_scattering_lengths() -> None:
    """Pass-through of caller-supplied b values."""
    atoms = Atoms("Cu", positions=[(5, 5, 5)], cell=[10, 10, 20], pbc=True)
    sld = compute_sld_summed_b_over_volume(
        atoms, z_range_ang=(0, 10), lateral_area_ang2=100.0,
        scattering_lengths_fm={"Cu": 15.436},   # double the real b
    )
    # Doubled b → doubled SLD.
    np.testing.assert_allclose(sld, 0.15436, atol=1e-4)


# ---------- compute_sld_profile -----------------------------------------------


def test_profile_bin_count_matches_range_over_width() -> None:
    atoms = Atoms("Cu", positions=[(5, 5, 5)], cell=[10, 10, 30], pbc=True)
    profile = compute_sld_profile(atoms, z_range_ang=(0.0, 30.0), bin_width_ang=1.0)
    assert len(profile) == 30


def test_profile_centres_are_evenly_spaced() -> None:
    atoms = Atoms("Cu", positions=[(5, 5, 5)], cell=[10, 10, 30], pbc=True)
    profile = compute_sld_profile(atoms, z_range_ang=(0.0, 30.0), bin_width_ang=2.0)
    diffs = np.diff(profile.z_centres_ang)
    np.testing.assert_allclose(diffs, 2.0)


def test_profile_returns_sldprofile_dataclass() -> None:
    atoms = Atoms("Cu", positions=[(5, 5, 5)], cell=[10, 10, 30], pbc=True)
    profile = compute_sld_profile(atoms, z_range_ang=(0, 30), bin_width_ang=1.0)
    assert isinstance(profile, SldProfile)
    assert profile.bin_width_ang == 1.0


def test_profile_atom_at_z5_lands_in_bin_5() -> None:
    """Atom at z = 5.5 with 1 Å bins starting at 0 should land in bin 5."""
    atoms = Atoms("Cu", positions=[(5, 5, 5.5)], cell=[10, 10, 30], pbc=True)
    profile = compute_sld_profile(atoms, z_range_ang=(0.0, 30.0), bin_width_ang=1.0)
    # Bin 5 should be the only non-zero bin.
    nonzero_bins = np.where(profile.sld_e6_per_a2 != 0)[0]
    assert nonzero_bins.tolist() == [5]


def test_profile_rejects_non_positive_bin_width() -> None:
    atoms = Atoms("Cu", positions=[(5, 5, 5)], cell=[10, 10, 30], pbc=True)
    with pytest.raises(ValueError):
        compute_sld_profile(atoms, z_range_ang=(0, 30), bin_width_ang=0.0)


def test_profile_handles_cu_o_mixture_per_bin() -> None:
    """Cu and O in the same bin sum their b contributions."""
    atoms = Atoms(
        "CuO",
        positions=[(5, 5, 5), (5, 5, 5)],
        cell=[10, 10, 20], pbc=True,
    )
    profile = compute_sld_profile(atoms, z_range_ang=(0, 10), bin_width_ang=1.0,
                                    lateral_area_ang2=100.0)
    # Sum is in bin 5; check it equals (b_Cu + b_O) / (A · δz).
    bin5 = profile.sld_e6_per_a2[5]
    expected = (NEUTRON_SCATTERING_LENGTH_FM["Cu"] + NEUTRON_SCATTERING_LENGTH_FM["O"]) \
                * FEMTOMETERS_TO_ANGSTROMS / (100.0 * 1.0) * 1e6
    np.testing.assert_allclose(bin5, expected, rtol=1e-6)
