"""Held-out test-set validation for a fine-tuned MACE potential.

After ``scripts/finetune_mace.sh`` produces a fine-tuned model, this
module evaluates it on the held-out test extxyz file and reports the
two metrics the project cares about:

- **Energy MAE per atom** (meV/atom). Target band:
  10–30 meV/atom (see :doc:`/docs/ml-gcgo-pivot.md` — manuscript reports
  9.8 meV/atom on PBEsol; we expect ~10–20 on PBE).
- **Force component MAE** (meV/Å). Target: <100 meV/Å.

The metric functions are pure NumPy and don't need MACE installed; the
``evaluate_model_on_extxyz`` end-to-end helper lazy-imports
:mod:`mace.calculators` so the test suite (which doesn't have MACE
here) only exercises the math.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms

ENERGY_MAE_TARGET_MEV_PER_ATOM = 30.0
"""Acceptance threshold for the fine-tuned model's energy MAE. Above this,
something in the pipeline is broken — :doc:`/docs/ml-gcgo-pivot.md` §3.1
attributes the value to functional miscalibration (PBE vs PBEsol)."""

FORCE_MAE_TARGET_MEV_PER_ANGSTROM = 100.0
"""Acceptance threshold for the fine-tuned model's force MAE."""


@dataclass(frozen=True)
class ValidationMetrics:
    """Test-set metrics for a fine-tuned MACE model.

    ``energy_mae_mev_per_atom`` is the project's headline number;
    ``force_mae_mev_per_angstrom`` is the secondary check. Both are
    reported in milli-units to match the manuscript's convention.
    """

    n_structures: int
    energy_mae_mev_per_atom: float
    force_mae_mev_per_angstrom: float

    def passes_targets(
        self,
        energy_threshold_mev_per_atom: float = ENERGY_MAE_TARGET_MEV_PER_ATOM,
        force_threshold_mev_per_angstrom: float = FORCE_MAE_TARGET_MEV_PER_ANGSTROM,
    ) -> bool:
        return (
            self.energy_mae_mev_per_atom <= energy_threshold_mev_per_atom
            and self.force_mae_mev_per_angstrom <= force_threshold_mev_per_angstrom
        )

    def summary(self) -> str:
        return (
            f"n={self.n_structures}  "
            f"E_MAE={self.energy_mae_mev_per_atom:.2f} meV/atom  "
            f"F_MAE={self.force_mae_mev_per_angstrom:.2f} meV/Å"
        )


def energy_mae_per_atom_mev(
    reference_ev: Sequence[float],
    predicted_ev: Sequence[float],
    n_atoms_per_structure: Sequence[int],
) -> float:
    """Energy MAE in meV/atom over a set of structures.

    Both ``reference_ev`` and ``predicted_ev`` must be ordered consistently
    with ``n_atoms_per_structure``. Each structure contributes
    ``|E_ref - E_pred| / N_atoms`` to the average.

    Args:
        reference_ev: Reference total energies (eV), one per structure.
        predicted_ev: Predicted total energies (eV), one per structure.
        n_atoms_per_structure: Atom counts, one per structure.

    Returns:
        Mean of per-structure absolute-error-per-atom, expressed in
        meV/atom.

    Raises:
        ValueError: If the three sequences have different lengths, the
            inputs are empty, or any atom count is non-positive.
    """
    ref = np.asarray(reference_ev, dtype=float)
    pred = np.asarray(predicted_ev, dtype=float)
    n = np.asarray(n_atoms_per_structure, dtype=int)

    _validate_lengths(ref, pred, n)
    if np.any(n <= 0):
        raise ValueError(f"n_atoms_per_structure must be positive; got {n}.")

    per_atom_abs_error_ev = np.abs(ref - pred) / n
    return float(per_atom_abs_error_ev.mean()) * 1000.0


def force_mae_mev_per_angstrom(
    reference_forces_ev_per_angstrom: Sequence[np.ndarray],
    predicted_forces_ev_per_angstrom: Sequence[np.ndarray],
) -> float:
    """Force MAE in meV/Å over all atom-component pairs in the test set.

    The standard MLIP convention: average absolute error across every
    (structure, atom, xyz-component) triple. Equivalent to flattening all
    force arrays and computing :func:`numpy.mean(|f_ref - f_pred|)`.

    Args:
        reference_forces_ev_per_angstrom: Reference force arrays (one
            ``(n_atoms, 3)`` per structure).
        predicted_forces_ev_per_angstrom: Predicted force arrays (same
            shape).

    Returns:
        Mean absolute force-component error, expressed in meV/Å.

    Raises:
        ValueError: If lengths differ or a pair has inconsistent shapes.
    """
    if len(reference_forces_ev_per_angstrom) != len(predicted_forces_ev_per_angstrom):
        raise ValueError(
            f"Reference/predicted force list length mismatch: "
            f"{len(reference_forces_ev_per_angstrom)} vs "
            f"{len(predicted_forces_ev_per_angstrom)}."
        )

    abs_errors: list[float] = []
    for ref, pred in zip(
        reference_forces_ev_per_angstrom, predicted_forces_ev_per_angstrom, strict=True
    ):
        ref_arr = np.asarray(ref, dtype=float)
        pred_arr = np.asarray(pred, dtype=float)
        if ref_arr.shape != pred_arr.shape:
            raise ValueError(
                f"Reference / predicted force array shape mismatch: "
                f"{ref_arr.shape} vs {pred_arr.shape}."
            )
        abs_errors.extend(np.abs(ref_arr - pred_arr).flatten().tolist())

    if not abs_errors:
        raise ValueError("No force entries to evaluate.")

    return float(np.mean(abs_errors)) * 1000.0


def compute_metrics_from_predictions(
    reference_structures: Sequence[Atoms],
    predicted_energies_ev: Sequence[float],
    predicted_forces_ev_per_angstrom: Sequence[np.ndarray],
) -> ValidationMetrics:
    """Bundle the two MAE numbers into a single :class:`ValidationMetrics`.

    The model-agnostic entry point: pass reference structures (with
    energy + forces attached) and matching predictions, get back the
    aggregated metrics. :func:`evaluate_model_on_extxyz` calls this
    after running the actual MACE forward pass.

    Args:
        reference_structures: ASE :class:`Atoms` with energy + forces
            attached (typically via the QE output read).
        predicted_energies_ev: One predicted total energy per structure.
        predicted_forces_ev_per_angstrom: One ``(n_atoms, 3)`` force
            array per structure.

    Returns:
        :class:`ValidationMetrics`.
    """
    if not reference_structures:
        raise ValueError("reference_structures is empty.")

    ref_energies = [float(s.get_potential_energy()) for s in reference_structures]
    ref_forces = [np.asarray(s.get_forces()) for s in reference_structures]
    n_atoms = [len(s) for s in reference_structures]

    e_mae = energy_mae_per_atom_mev(ref_energies, predicted_energies_ev, n_atoms)
    f_mae = force_mae_mev_per_angstrom(ref_forces, predicted_forces_ev_per_angstrom)

    return ValidationMetrics(
        n_structures=len(reference_structures),
        energy_mae_mev_per_atom=e_mae,
        force_mae_mev_per_angstrom=f_mae,
    )


def evaluate_model_on_extxyz(
    model_path: str | os.PathLike[str],
    test_extxyz_path: str | os.PathLike[str],
    *,
    device: str = "cuda",
) -> ValidationMetrics:
    """End-to-end: load MACE model, predict on the test set, return metrics.

    Lazy-imports :mod:`mace.calculators` and :mod:`ase.io`, so the
    surrounding module stays importable without the ML extras.

    Args:
        model_path: Path to the fine-tuned ``.model`` file produced by
            ``mace_run_train``.
        test_extxyz_path: Path to the test extxyz file used during
            fine-tuning (manuscript uses the same 1/11 hold-out).
        device: ``"cuda"`` on DGX Spark, ``"cpu"`` for ad-hoc checks.

    Returns:
        :class:`ValidationMetrics`.

    Raises:
        ImportError: If :mod:`mace` is not installed.
        FileNotFoundError: If either path does not resolve to a file.
    """
    model_path = Path(model_path)
    test_extxyz_path = Path(test_extxyz_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not test_extxyz_path.is_file():
        raise FileNotFoundError(f"Test extxyz not found: {test_extxyz_path}")

    from ase.io import read as ase_read  # lazy
    from mace.calculators import MACECalculator  # lazy

    test_structures = ase_read(str(test_extxyz_path), index=":")
    if isinstance(test_structures, Atoms):
        test_structures = [test_structures]

    calc = MACECalculator(model_paths=[str(model_path)], device=device)

    predicted_energies: list[float] = []
    predicted_forces: list[np.ndarray] = []
    for atoms in test_structures:
        eval_atoms = atoms.copy()
        eval_atoms.calc = calc
        predicted_energies.append(float(eval_atoms.get_potential_energy()))
        predicted_forces.append(np.asarray(eval_atoms.get_forces()))

    return compute_metrics_from_predictions(
        test_structures, predicted_energies, predicted_forces
    )


def _validate_lengths(*arrays: np.ndarray) -> None:
    lengths = [a.shape[0] for a in arrays]
    if len(set(lengths)) != 1:
        raise ValueError(f"Input lengths must match; got {lengths}.")
    if lengths[0] == 0:
        raise ValueError("Inputs are empty; nothing to evaluate.")
