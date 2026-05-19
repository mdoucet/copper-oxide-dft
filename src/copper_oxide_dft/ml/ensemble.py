"""Post-process the GCGA output into a per-x_O minimum-Ω ensemble.

After :func:`copper_oxide_dft.ml.gcga.run_gcga_sweep` runs at multiple
μ_O values (and a biased pass), the ensemble of all final populations
is reduced to a single curve: the lowest grand-potential structure
*per x_O bin*. That curve is what gets handed to Block F
(:mod:`copper_oxide_dft.ml.fcp_rerank`) for the constant-potential
ESM-FCP rerank.

Storage convention:

- In-memory: a list of :class:`Phase` dataclasses.
- On disk: an extended-XYZ file per ensemble, with the metadata
  (``mu_o_ev``, ``omega_o_ev``, ``x_o``, ``source``) embedded in each
  frame's ``info`` dict so the round-trip is lossless.

Reading the ensemble back lazily-imports :mod:`ase.io`. Writing uses
the same.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ase import Atoms

DEFAULT_N_X_O_BINS = 20
"""How many bins along x_O ∈ [0, 1] to coarsen the ensemble to.
Manuscript-comparable resolution; downstream Frontier rerank picks the
top-K across these bins."""


@dataclass(frozen=True)
class Phase:
    """One GCGA ensemble member.

    Attributes:
        atoms: ASE structure (with energy attached via SinglePointCalculator).
        energy_ev: MACE-predicted total energy of the structure (eV).
        mu_o_ev: O chemical potential the search was run at (eV vs vacuum).
        x_o: Stoichiometry (N_O / (N_Cu + N_O)).
        omega_o_ev: Grand potential ``E - μ_O · N_O`` (eV).
        source: ``"unbiased"`` or ``"biased"``; useful when diagnosing
            why a particular x_O bin is or isn't represented.
        index_in_source: Position in the originating GCGA population
            (helps trace back when something looks wrong).
    """

    atoms: Atoms
    energy_ev: float
    mu_o_ev: float
    x_o: float
    omega_o_ev: float
    source: str
    index_in_source: int = 0


def phase_from_atoms(
    atoms: Atoms,
    energy_ev: float,
    mu_o_ev: float,
    source: str,
    *,
    index_in_source: int = 0,
) -> Phase:
    """Build a :class:`Phase` from a structure + energy + (mu_O, source).

    Computes x_O and the grand potential internally so callers don't have
    to repeat the arithmetic. Used by both the GCGA driver (to record
    finished candidates) and the test suite.
    """
    from copper_oxide_dft.ml.gcga import compute_x_o, grand_potential_ev

    return Phase(
        atoms=atoms,
        energy_ev=float(energy_ev),
        mu_o_ev=float(mu_o_ev),
        x_o=compute_x_o(atoms),
        omega_o_ev=grand_potential_ev(energy_ev, atoms, mu_o_ev),
        source=source,
        index_in_source=int(index_in_source),
    )


def merge_ensembles(*ensembles: Iterable[Phase]) -> list[Phase]:
    """Concatenate ensembles, drop ω duplicates.

    Two phases are "duplicates" if they share the same chemical formula
    and their ω values differ by less than 1 meV — that's tighter than
    the MACE test MAE (~10 meV/atom on a ~100-atom cell ≈ 1 eV total),
    so this is a conservative dedup that won't merge genuinely distinct
    isomers.

    Args:
        *ensembles: Iterables of :class:`Phase`.

    Returns:
        Deduplicated list, sorted by ``(x_o, omega_o_ev)`` for stable
        downstream consumers.
    """
    all_phases: list[Phase] = []
    for batch in ensembles:
        all_phases.extend(batch)
    if not all_phases:
        return []

    # Dedup by (chemical formula, rounded omega).
    seen: set[tuple[str, int]] = set()
    deduped: list[Phase] = []
    for phase in all_phases:
        key = (
            phase.atoms.get_chemical_formula(),
            round(phase.omega_o_ev * 1000.0),  # meV bucket
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(phase)

    deduped.sort(key=lambda p: (p.x_o, p.omega_o_ev))
    return deduped


def per_x_o_minima(
    phases: Sequence[Phase],
    *,
    n_bins: int = DEFAULT_N_X_O_BINS,
    x_o_range: tuple[float, float] = (0.0, 1.0),
) -> list[Phase]:
    """For each x_O bin, return the phase with the lowest ω_O.

    The headline output of the ensemble step. ``x_o_range`` defaults to
    the full [0, 1] interval; pass a narrower range if you want to
    restrict to (say) the experimentally relevant suboxide window.

    Args:
        phases: Merged ensemble.
        n_bins: Number of equal-width bins across ``x_o_range``.
        x_o_range: Inclusive (low, high) edges for binning.

    Returns:
        Up to ``n_bins`` phases (some bins may be empty), sorted by x_O.

    Raises:
        ValueError: If ``n_bins < 1`` or ``x_o_range`` is degenerate.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1; got {n_bins}.")
    low, high = x_o_range
    if not low < high:
        raise ValueError(f"x_o_range must have low < high; got {x_o_range}.")
    if not phases:
        return []

    bin_width = (high - low) / n_bins
    chosen: dict[int, Phase] = {}
    for phase in phases:
        if not low <= phase.x_o <= high:
            continue
        bin_idx = min(int((phase.x_o - low) / bin_width), n_bins - 1)
        current = chosen.get(bin_idx)
        if current is None or phase.omega_o_ev < current.omega_o_ev:
            chosen[bin_idx] = phase

    return sorted(chosen.values(), key=lambda p: p.x_o)


def top_k_by_omega(
    phases: Sequence[Phase],
    k: int,
) -> list[Phase]:
    """Return the k phases with the lowest Ω.

    Used to pick the candidates for the Block F ESM-FCP rerank on
    Frontier (typically K = 20).

    Args:
        phases: Ensemble (already merged + per-x_O coarsened or not).
        k: How many to keep.

    Returns:
        Up to ``k`` phases, sorted by ascending Ω.
    """
    if k < 0:
        raise ValueError(f"k must be non-negative; got {k}.")
    return sorted(phases, key=lambda p: p.omega_o_ev)[:k]


def write_ensemble_extxyz(
    phases: Sequence[Phase],
    out_path: str | os.PathLike[str],
) -> Path:
    """Persist an ensemble as a single extxyz file, metadata in ``info``.

    Each frame carries:

    - ``info["mu_o_ev"]``, ``info["x_o"]``, ``info["omega_o_ev"]``,
      ``info["source"]``, ``info["index_in_source"]``.
    - The SinglePointCalculator's ``energy`` (so ASE round-trips fine).

    Args:
        phases: List of :class:`Phase` to write.
        out_path: Destination ``.extxyz`` path.

    Returns:
        Resolved path of the written file.
    """
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import write as ase_write

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames: list[Atoms] = []
    for phase in phases:
        frame = phase.atoms.copy()
        # Preserve forces if the original calc had them; otherwise set
        # energy-only so the extxyz writer doesn't drop the energy.
        existing_forces = None
        if phase.atoms.calc is not None:
            try:
                existing_forces = phase.atoms.get_forces()
            except Exception:  # noqa: BLE001
                existing_forces = None
        frame.calc = SinglePointCalculator(
            frame,
            energy=phase.energy_ev,
            forces=existing_forces,
        )
        frame.info.update(
            {
                "mu_o_ev": phase.mu_o_ev,
                "x_o": phase.x_o,
                "omega_o_ev": phase.omega_o_ev,
                "source": phase.source,
                "index_in_source": phase.index_in_source,
            }
        )
        frames.append(frame)
    ase_write(str(out_path), frames, format="extxyz")
    return out_path


def read_ensemble_extxyz(in_path: str | os.PathLike[str]) -> list[Phase]:
    """Inverse of :func:`write_ensemble_extxyz`.

    Reconstructs the :class:`Phase` list from an extxyz file. The
    ``info`` keys ``mu_o_ev`` / ``x_o`` / ``omega_o_ev`` / ``source``
    are required; missing keys raise.

    Args:
        in_path: Path written by :func:`write_ensemble_extxyz`.

    Returns:
        List of :class:`Phase` in file order.
    """
    from ase.io import read as ase_read

    in_path = Path(in_path)
    if in_path.stat().st_size == 0:
        return []
    frames = ase_read(str(in_path), index=":")
    if isinstance(frames, Atoms):
        frames = [frames]

    phases: list[Phase] = []
    for frame in frames:
        info = frame.info
        if "mu_o_ev" not in info or "omega_o_ev" not in info:
            raise ValueError(
                f"Ensemble extxyz frame missing required keys in info: {info}"
            )
        energy = (
            float(frame.get_potential_energy())
            if frame.calc is not None
            else float(info.get("energy_ev", 0.0))
        )
        phases.append(
            Phase(
                atoms=frame,
                energy_ev=energy,
                mu_o_ev=float(info["mu_o_ev"]),
                x_o=float(info["x_o"]) if "x_o" in info else _compute_x_o_local(frame),
                omega_o_ev=float(info["omega_o_ev"]),
                source=str(info.get("source", "unknown")),
                index_in_source=int(info.get("index_in_source", 0)),
            )
        )
    return phases


def _compute_x_o_local(atoms: Atoms) -> float:
    """Fallback used when an extxyz frame lacks an explicit x_o info key."""
    from copper_oxide_dft.ml.gcga import compute_x_o

    return compute_x_o(atoms)
