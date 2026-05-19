"""Tests for copper_oxide_dft.ml.validate.

The pure-numpy MAE math is covered fully here. The MACE end-to-end
helper (:func:`evaluate_model_on_extxyz`) is exercised only by an
ImportError smoke test — actually invoking MACE needs the model + GPU,
which lives on DGX Spark.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from copper_oxide_dft.ml.validate import (
    ENERGY_MAE_TARGET_MEV_PER_ATOM,
    FORCE_MAE_TARGET_MEV_PER_ANGSTROM,
    ValidationMetrics,
    compute_metrics_from_predictions,
    energy_mae_per_atom_mev,
    evaluate_model_on_extxyz,
    force_mae_mev_per_angstrom,
)

# ---------- energy_mae_per_atom_mev ------------------------------------------


def test_energy_mae_perfect_prediction_is_zero() -> None:
    assert energy_mae_per_atom_mev([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [10, 20, 30]) == 0.0


def test_energy_mae_known_value() -> None:
    # Errors: 0.1, 0.1, 0.1 eV across structures with 100, 100, 100 atoms.
    # Per-atom errors: 0.001 eV = 1 meV/atom. Mean = 1 meV/atom.
    mae = energy_mae_per_atom_mev([10.0, 20.0, 30.0], [10.1, 19.9, 30.1], [100, 100, 100])
    np.testing.assert_allclose(mae, 1.0, atol=1e-9)


def test_energy_mae_weights_by_atom_count() -> None:
    """Larger structures should get *smaller* per-atom errors for the same total error."""
    mae_small = energy_mae_per_atom_mev([1.0], [1.1], [1])      # 0.1 eV / 1 atom = 100 meV/atom
    mae_large = energy_mae_per_atom_mev([1.0], [1.1], [100])    # 0.1 eV / 100 atoms = 1 meV/atom
    assert mae_small > mae_large
    np.testing.assert_allclose(mae_small / mae_large, 100.0, rtol=1e-9)


def test_energy_mae_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        energy_mae_per_atom_mev([1.0, 2.0], [1.0], [10, 20])


def test_energy_mae_rejects_empty() -> None:
    with pytest.raises(ValueError):
        energy_mae_per_atom_mev([], [], [])


def test_energy_mae_rejects_zero_atom_count() -> None:
    with pytest.raises(ValueError):
        energy_mae_per_atom_mev([1.0], [1.0], [0])
    with pytest.raises(ValueError):
        energy_mae_per_atom_mev([1.0], [1.0], [-5])


# ---------- force_mae_mev_per_angstrom ----------------------------------------


def test_force_mae_perfect_prediction_is_zero() -> None:
    ref = [np.array([[1.0, 2.0, 3.0]])]
    mae = force_mae_mev_per_angstrom(ref, [r.copy() for r in ref])
    assert mae == 0.0


def test_force_mae_known_value() -> None:
    # One structure, 2 atoms, 3 components each = 6 entries.
    # Errors: 0.01 eV/Å on every component. MAE = 10 meV/Å.
    ref = [np.zeros((2, 3))]
    pred = [np.full((2, 3), 0.01)]
    mae = force_mae_mev_per_angstrom(ref, pred)
    np.testing.assert_allclose(mae, 10.0, atol=1e-9)


def test_force_mae_averages_across_structures_proportional_to_size() -> None:
    """Structures of different sizes contribute their atom-component counts."""
    # Structure A: 1 atom (3 components), all errors 0.01 eV/Å → contributes 3 entries of 10 meV/Å
    # Structure B: 10 atoms (30 components), all errors 0.10 eV/Å → contributes 30 entries of 100 meV/Å
    # Weighted mean: (3 * 10 + 30 * 100) / 33 ≈ 91.8 meV/Å
    ref_a = np.zeros((1, 3))
    pred_a = np.full((1, 3), 0.01)
    ref_b = np.zeros((10, 3))
    pred_b = np.full((10, 3), 0.10)
    mae = force_mae_mev_per_angstrom([ref_a, ref_b], [pred_a, pred_b])
    expected = (3 * 10.0 + 30 * 100.0) / 33.0
    np.testing.assert_allclose(mae, expected, rtol=1e-6)


def test_force_mae_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        force_mae_mev_per_angstrom([np.zeros((1, 3))], [np.zeros((1, 3)), np.zeros((1, 3))])


def test_force_mae_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        force_mae_mev_per_angstrom([np.zeros((2, 3))], [np.zeros((3, 3))])


def test_force_mae_rejects_empty_inputs() -> None:
    with pytest.raises(ValueError):
        force_mae_mev_per_angstrom([], [])


# ---------- compute_metrics_from_predictions ----------------------------------


def _atoms_with_e_f(energy_ev: float, forces: np.ndarray, n_atoms: int) -> Atoms:
    atoms = Atoms(
        symbols="Cu" * n_atoms,
        positions=np.zeros((n_atoms, 3)),
        cell=[10, 10, 10],
        pbc=True,
    )
    atoms.calc = SinglePointCalculator(atoms, energy=energy_ev, forces=forces)
    return atoms


def test_compute_metrics_bundles_both_maes() -> None:
    structs = [
        _atoms_with_e_f(10.0, np.zeros((2, 3)), 2),
        _atoms_with_e_f(20.0, np.zeros((3, 3)), 3),
    ]
    predicted_energies = [10.002, 20.003]  # 1 and 1 meV/atom respectively
    predicted_forces = [np.full((2, 3), 0.001), np.full((3, 3), 0.001)]  # all 1 meV/Å

    metrics = compute_metrics_from_predictions(structs, predicted_energies, predicted_forces)
    assert isinstance(metrics, ValidationMetrics)
    assert metrics.n_structures == 2
    np.testing.assert_allclose(metrics.energy_mae_mev_per_atom, 1.0, atol=1e-3)
    np.testing.assert_allclose(metrics.force_mae_mev_per_angstrom, 1.0, atol=1e-3)


def test_compute_metrics_rejects_empty_reference() -> None:
    with pytest.raises(ValueError):
        compute_metrics_from_predictions([], [], [])


# ---------- ValidationMetrics behavior ---------------------------------------


def test_metrics_summary_includes_both_numbers() -> None:
    m = ValidationMetrics(n_structures=42, energy_mae_mev_per_atom=12.3,
                           force_mae_mev_per_angstrom=45.6)
    s = m.summary()
    assert "42" in s
    assert "12.3" in s
    assert "45.6" in s
    assert "meV/atom" in s


def test_metrics_passes_targets_inside_envelope() -> None:
    m = ValidationMetrics(n_structures=1, energy_mae_mev_per_atom=15.0,
                           force_mae_mev_per_angstrom=50.0)
    assert m.passes_targets() is True


def test_metrics_fails_when_energy_above_threshold() -> None:
    m = ValidationMetrics(n_structures=1,
                           energy_mae_mev_per_atom=ENERGY_MAE_TARGET_MEV_PER_ATOM + 1.0,
                           force_mae_mev_per_angstrom=50.0)
    assert m.passes_targets() is False


def test_metrics_fails_when_force_above_threshold() -> None:
    m = ValidationMetrics(n_structures=1, energy_mae_mev_per_atom=15.0,
                           force_mae_mev_per_angstrom=FORCE_MAE_TARGET_MEV_PER_ANGSTROM + 1.0)
    assert m.passes_targets() is False


def test_thresholds_match_pivot_doc() -> None:
    """The targets in ml-gcgo-pivot.md are 30 meV/atom and 100 meV/Å."""
    assert ENERGY_MAE_TARGET_MEV_PER_ATOM == 30.0
    assert FORCE_MAE_TARGET_MEV_PER_ANGSTROM == 100.0


# ---------- evaluate_model_on_extxyz (end-to-end, MACE-dependent) ------------


def test_evaluate_model_on_extxyz_raises_when_model_missing(tmp_path: Path) -> None:
    test_extxyz = tmp_path / "test.extxyz"
    test_extxyz.write_text("")
    with pytest.raises(FileNotFoundError):
        evaluate_model_on_extxyz(tmp_path / "does-not-exist.model", test_extxyz)


def test_evaluate_model_on_extxyz_raises_when_test_extxyz_missing(tmp_path: Path) -> None:
    model = tmp_path / "fake.model"
    model.write_bytes(b"")
    with pytest.raises(FileNotFoundError):
        evaluate_model_on_extxyz(model, tmp_path / "does-not-exist.extxyz")
