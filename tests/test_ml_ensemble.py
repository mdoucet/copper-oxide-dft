"""Tests for copper_oxide_dft.ml.ensemble."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from copper_oxide_dft.ml.ensemble import (
    DEFAULT_N_X_O_BINS,
    Phase,
    merge_ensembles,
    per_x_o_minima,
    phase_from_atoms,
    read_ensemble_extxyz,
    top_k_by_omega,
    write_ensemble_extxyz,
)
from copper_oxide_dft.structure_builder import (
    build_bulk_cu,
    build_bulk_cu2o,
    build_bulk_cuo,
)

# ---------- phase_from_atoms --------------------------------------------------


def test_phase_from_atoms_computes_x_o_and_omega() -> None:
    cu2o = build_bulk_cu2o()
    phase = phase_from_atoms(cu2o, energy_ev=-100.0, mu_o_ev=-6.0, source="unbiased")
    np.testing.assert_allclose(phase.x_o, 2.0 / 6.0)
    # Ω = E - μ_O · N_O = -100 - (-6 · 2) = -88
    np.testing.assert_allclose(phase.omega_o_ev, -88.0)
    assert phase.source == "unbiased"


def test_phase_from_atoms_records_source_index() -> None:
    phase = phase_from_atoms(build_bulk_cu(), -1.0, -6.0, "biased", index_in_source=42)
    assert phase.index_in_source == 42


# ---------- merge_ensembles ---------------------------------------------------


def test_merge_empty_returns_empty() -> None:
    assert merge_ensembles() == []
    assert merge_ensembles([], []) == []


def test_merge_concatenates_distinct_phases() -> None:
    p1 = phase_from_atoms(build_bulk_cu(), -1.0, -6.0, "unbiased")
    p2 = phase_from_atoms(build_bulk_cu2o(), -100.0, -6.0, "biased")
    merged = merge_ensembles([p1], [p2])
    assert len(merged) == 2


def test_merge_dedupes_within_a_milli_ev_of_omega() -> None:
    """Same formula + same ω (to 1 meV) → dedup."""
    cu2o_a = build_bulk_cu2o()
    cu2o_b = build_bulk_cu2o()
    p1 = phase_from_atoms(cu2o_a, -100.0000, -6.0, "unbiased")
    p2 = phase_from_atoms(cu2o_b, -100.0003, -6.0, "biased")  # ω within 1 meV
    merged = merge_ensembles([p1, p2])
    assert len(merged) == 1


def test_merge_keeps_phases_with_different_formulas() -> None:
    """Same ω but different composition shouldn't merge."""
    p1 = phase_from_atoms(build_bulk_cu(), -1.0, -6.0, "unbiased")
    p2 = phase_from_atoms(build_bulk_cu2o(), -1.0, -6.0, "biased")
    merged = merge_ensembles([p1, p2])
    assert len(merged) == 2


def test_merge_sorts_by_x_o_then_omega() -> None:
    p_low_x_high_omega = phase_from_atoms(build_bulk_cu(), -1.0, -6.0, "u")
    p_high_x_low_omega = phase_from_atoms(build_bulk_cuo(), -200.0, -6.0, "u")
    merged = merge_ensembles([p_high_x_low_omega], [p_low_x_high_omega])
    assert merged[0].x_o < merged[1].x_o


# ---------- per_x_o_minima ----------------------------------------------------


def test_per_x_o_minima_picks_lowest_omega_per_bin() -> None:
    cu2o_a = build_bulk_cu2o()
    cu2o_b = build_bulk_cu2o()
    # Same x_O (1/3), different ω. Should pick the one with lower ω.
    p_high = phase_from_atoms(cu2o_a, energy_ev=-100.0, mu_o_ev=-6.0, source="u")
    p_low = phase_from_atoms(cu2o_b, energy_ev=-105.0, mu_o_ev=-6.0, source="u")
    minima = per_x_o_minima([p_high, p_low], n_bins=10)
    assert len(minima) == 1
    assert minima[0].omega_o_ev == p_low.omega_o_ev


def test_per_x_o_minima_returns_one_phase_per_occupied_bin() -> None:
    phases = [
        phase_from_atoms(build_bulk_cu(), -1.0, -6.0, "u"),       # x_O = 0.0
        phase_from_atoms(build_bulk_cu2o(), -100.0, -6.0, "u"),   # x_O ≈ 0.333
        phase_from_atoms(build_bulk_cuo(), -200.0, -6.0, "u"),    # x_O = 0.5
    ]
    minima = per_x_o_minima(phases, n_bins=10)
    assert len(minima) == 3
    assert sorted(p.x_o for p in minima) == [0.0, 2.0 / 6.0, 0.5]


def test_per_x_o_minima_filters_to_range() -> None:
    phases = [
        phase_from_atoms(build_bulk_cu(), -1.0, -6.0, "u"),       # 0.0
        phase_from_atoms(build_bulk_cu2o(), -100.0, -6.0, "u"),   # 0.333
        phase_from_atoms(build_bulk_cuo(), -200.0, -6.0, "u"),    # 0.5
    ]
    # Range that excludes both endpoints.
    minima = per_x_o_minima(phases, n_bins=5, x_o_range=(0.2, 0.4))
    assert len(minima) == 1
    np.testing.assert_allclose(minima[0].x_o, 2.0 / 6.0)


def test_per_x_o_minima_returns_sorted() -> None:
    phases = [
        phase_from_atoms(build_bulk_cuo(), -200.0, -6.0, "u"),
        phase_from_atoms(build_bulk_cu(), -1.0, -6.0, "u"),
        phase_from_atoms(build_bulk_cu2o(), -100.0, -6.0, "u"),
    ]
    minima = per_x_o_minima(phases, n_bins=10)
    assert [m.x_o for m in minima] == sorted(m.x_o for m in minima)


def test_per_x_o_minima_empty_input() -> None:
    assert per_x_o_minima([], n_bins=10) == []


def test_per_x_o_minima_rejects_bad_args() -> None:
    with pytest.raises(ValueError):
        per_x_o_minima([], n_bins=0)
    with pytest.raises(ValueError):
        per_x_o_minima([], n_bins=10, x_o_range=(0.5, 0.5))


def test_default_x_o_bins_matches_manuscript_resolution() -> None:
    assert DEFAULT_N_X_O_BINS == 20


# ---------- top_k_by_omega ----------------------------------------------------


def test_top_k_returns_lowest_omega_first() -> None:
    phases = [
        phase_from_atoms(build_bulk_cu2o(), e, -6.0, "u")
        for e in (-50.0, -100.0, -75.0)
    ]
    top = top_k_by_omega(phases, k=2)
    assert top[0].energy_ev == -100.0
    assert top[1].energy_ev == -75.0


def test_top_k_zero_returns_empty() -> None:
    phases = [phase_from_atoms(build_bulk_cu(), -1.0, -6.0, "u")]
    assert top_k_by_omega(phases, k=0) == []


def test_top_k_more_than_available_returns_all() -> None:
    phases = [phase_from_atoms(build_bulk_cu(), -1.0, -6.0, "u")]
    assert len(top_k_by_omega(phases, k=10)) == 1


def test_top_k_rejects_negative() -> None:
    with pytest.raises(ValueError):
        top_k_by_omega([], k=-1)


# ---------- write/read_ensemble_extxyz round-trip ----------------------------


def _phase_with_calc(atoms: Atoms, energy: float, mu_o: float, source: str) -> Phase:
    atoms_with_calc = atoms.copy()
    atoms_with_calc.calc = SinglePointCalculator(atoms_with_calc, energy=energy)
    return phase_from_atoms(atoms_with_calc, energy, mu_o, source)


def test_round_trip_preserves_metadata(tmp_path: Path) -> None:
    phases = [
        _phase_with_calc(build_bulk_cu(), -1.0, -6.5, "unbiased"),
        _phase_with_calc(build_bulk_cu2o(), -100.0, -6.5, "biased"),
        _phase_with_calc(build_bulk_cuo(), -200.0, -6.5, "biased"),
    ]
    out = tmp_path / "ens.extxyz"
    write_ensemble_extxyz(phases, out)
    roundtrip = read_ensemble_extxyz(out)
    assert len(roundtrip) == 3
    for original, restored in zip(phases, roundtrip, strict=True):
        assert original.source == restored.source
        np.testing.assert_allclose(original.mu_o_ev, restored.mu_o_ev)
        np.testing.assert_allclose(original.x_o, restored.x_o)
        np.testing.assert_allclose(original.omega_o_ev, restored.omega_o_ev)


def test_round_trip_preserves_chemical_formula(tmp_path: Path) -> None:
    phases = [_phase_with_calc(build_bulk_cuo(), -200.0, -6.5, "u")]
    out = tmp_path / "ens.extxyz"
    write_ensemble_extxyz(phases, out)
    restored = read_ensemble_extxyz(out)
    assert restored[0].atoms.get_chemical_formula() == "Cu4O4"


def test_write_then_read_empty_list(tmp_path: Path) -> None:
    out = tmp_path / "empty.extxyz"
    write_ensemble_extxyz([], out)
    assert read_ensemble_extxyz(out) == []
