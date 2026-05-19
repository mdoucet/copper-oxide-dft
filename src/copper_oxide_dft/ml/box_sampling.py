"""Box-sampling perturbation pipeline for the MLIP-GCGO dataset.

Generates diverse, non-stoichiometric Cu/Cu-O structures from a handful of
seed bulk cells (Cu, Cu2O, CuO, c-CuO). Each perturbation is one independent
sample for the DFT ground-truth dataset that fine-tunes MACE-MP-0.

The pipeline follows :doc:`/docs/machine-learned-dft.md` §2:

1. **Random rattle** displaces every atom by a Gaussian noise (default σ = 0.2 Å).
2. **Isotropic lattice scaling** multiplies the cell and fractional positions by
   a random scale factor (default ±5 %).
3. **Random O insert / delete** changes the stoichiometry. Insertions are
   attempted at random fractional positions; rejected if too close to an
   existing atom or if Cu-O connectivity cannot be established.
4. **Hookean repair** pushes apart any pair that ended up closer than a
   pair-specific cutoff. This is *not* a full geometry optimization — it's a
   cheap pre-opt so QE doesn't immediately diverge.
5. **Cu-O connectivity filter** rejects structures where any O atom has no
   Cu neighbour within ``max_cu_o_dist`` (default 2.8 Å). This matches the
   manuscript's "enforcing Cu-O connectivity" clause and prevents floating
   O atoms that don't represent a physical Cu-O environment.

This is a pure-ASE implementation — the perturbation step has no GA-backend
dependency. The downstream GCGA (Block E of the pivot plan) uses ase-ga; see
:mod:`copper_oxide_dft.ml.gcga`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ase import Atom, Atoms

CU_O_CONNECTIVITY_CUTOFF_ANG = 2.8
"""Maximum Cu-O distance considered a "bond" for connectivity filtering (Å).
Slightly larger than the typical Cu-O bond length in Cu oxides (~1.85-2.00 Å)
to tolerate rattled structures."""

DEFAULT_MIN_PAIR_DISTANCE_ANG: dict[frozenset[str], float] = {
    frozenset(["Cu", "Cu"]): 1.8,
    frozenset(["Cu", "O"]): 1.4,
    frozenset(["O", "O"]): 1.0,
}
"""Pair-specific minimum distances (Å). Below these, the Hookean repair pushes
the atoms apart. Values chosen well below typical bond lengths but above
nuclear-collision territory so the DFT SCF stays stable."""

DEFAULT_INSERT_ATTEMPTS = 50
"""How many random positions to try before giving up on a single insertion."""

DEFAULT_INSERT_MIN_DISTANCE_ANG = 1.6
"""Minimum distance to any existing atom when inserting a new O (Å). Tighter
than the Hookean cutoff because here we have full control over the position."""


@dataclass(frozen=True)
class BoxSamplingConfig:
    """All knobs for a box-sampling pass.

    Defaults match the manuscript walkthrough for *bulk* sampling. For
    surface slabs the manuscript uses smaller insert/delete budgets (up to
    2 Cu and 2 O per cell, vs up to 8 O for bulk); see :doc:`/docs/machine-learned-dft.md`.
    """

    rattle_stdev_ang: float = 0.2
    """Gaussian noise standard deviation applied to atomic positions (Å)."""

    lattice_scale: float = 0.05
    """Isotropic cell scale factor sampled uniformly from [1-s, 1+s]."""

    max_o_insertions: int = 8
    """Upper bound on the number of O atoms that can be inserted in one sample."""

    max_o_deletions: int = 8
    """Upper bound on the number of O atoms that can be deleted in one sample."""

    insert_min_distance_ang: float = DEFAULT_INSERT_MIN_DISTANCE_ANG
    """Minimum distance from existing atoms when attempting an insertion (Å)."""

    insert_attempts: int = DEFAULT_INSERT_ATTEMPTS
    """Random-position tries per insertion before giving up."""

    cu_o_connectivity_cutoff_ang: float = CU_O_CONNECTIVITY_CUTOFF_ANG
    """Max Cu-O distance for the connectivity filter (Å)."""

    hookean_max_steps: int = 20
    """How many repair iterations the Hookean step is allowed."""

    min_pair_distance_ang: dict[frozenset[str], float] = field(
        default_factory=lambda: dict(DEFAULT_MIN_PAIR_DISTANCE_ANG)
    )
    """Pair-specific minimum distances (Å) below which atoms get pushed apart."""

    enforce_connectivity: bool = True
    """If True, reject samples that fail the Cu-O connectivity check."""


@dataclass(frozen=True)
class PerturbationResult:
    """Outcome of a single perturbation attempt.

    ``atoms`` is None if the perturbation could not produce a structure that
    passed the connectivity filter (after the configured number of repair
    iterations). ``info`` records what happened, so a manifest can report
    why samples were rejected.
    """

    atoms: Atoms | None
    info: dict[str, object]

    @property
    def accepted(self) -> bool:
        return self.atoms is not None


def perturb_structure(
    seed: Atoms,
    config: BoxSamplingConfig,
    rng: np.random.Generator,
) -> PerturbationResult:
    """Produce one perturbed structure from a seed cell.

    Args:
        seed: Bulk or slab Atoms to perturb. Not modified.
        config: Perturbation knobs (rattle stdev, scale, insert/delete limits).
        rng: NumPy random generator (pass an explicit one for reproducibility).

    Returns:
        A :class:`PerturbationResult`. ``result.accepted`` is True if the
        perturbation passed all filters; ``result.atoms`` is the new structure.

    Example:
        >>> from ase.build import bulk
        >>> rng = np.random.default_rng(0)
        >>> seed = bulk("Cu", "fcc", a=3.6, cubic=True)
        >>> result = perturb_structure(seed, BoxSamplingConfig(), rng)
        >>> result.accepted
        True
        >>> result.atoms is not seed   # always a fresh copy
        True
    """
    atoms = seed.copy()

    info: dict[str, object] = {}

    scale = float(1.0 + rng.uniform(-config.lattice_scale, config.lattice_scale))
    atoms.set_cell(atoms.cell.array * scale, scale_atoms=True)
    info["lattice_scale"] = scale

    if config.rattle_stdev_ang > 0:
        atoms.rattle(stdev=config.rattle_stdev_ang, seed=int(rng.integers(0, 2**31 - 1)))
    info["rattle_stdev_ang"] = config.rattle_stdev_ang

    n_delete = int(rng.integers(0, config.max_o_deletions + 1))
    deleted = _delete_random_oxygens(atoms, n_delete, rng)
    info["o_deleted"] = deleted

    n_insert = int(rng.integers(0, config.max_o_insertions + 1))
    inserted = _insert_random_oxygens(atoms, n_insert, config, rng)
    info["o_inserted"] = inserted

    repair_steps = apply_hookean_repair(atoms, config)
    info["repair_steps"] = repair_steps

    connectivity_ok = enforce_cu_o_connectivity(atoms, config.cu_o_connectivity_cutoff_ang)
    info["cu_o_connectivity_ok"] = connectivity_ok

    if config.enforce_connectivity and not connectivity_ok:
        return PerturbationResult(atoms=None, info=info)

    return PerturbationResult(atoms=atoms, info=info)


def sample_batch(
    seed: Atoms,
    n_samples: int,
    config: BoxSamplingConfig,
    rng: np.random.Generator,
    max_attempts_per_sample: int = 10,
) -> list[PerturbationResult]:
    """Generate ``n_samples`` perturbations of a seed cell.

    Rejected attempts (failing the connectivity filter) are retried up to
    ``max_attempts_per_sample`` times. Returns one :class:`PerturbationResult`
    per requested sample; if even the retries cannot produce a valid
    structure, the result has ``atoms=None`` so the caller can decide whether
    to escalate.

    Args:
        seed: Base cell to perturb (not mutated).
        n_samples: How many accepted samples to aim for.
        config: Box-sampling configuration.
        rng: NumPy generator.
        max_attempts_per_sample: Retry budget per sample slot.

    Returns:
        List of length ``n_samples``.
    """
    if n_samples < 0:
        raise ValueError(f"n_samples must be non-negative; got {n_samples}.")
    if max_attempts_per_sample < 1:
        raise ValueError(
            f"max_attempts_per_sample must be >= 1; got {max_attempts_per_sample}."
        )

    results: list[PerturbationResult] = []
    for _ in range(n_samples):
        attempt_info: dict[str, object] = {"attempts": 0}
        result = PerturbationResult(atoms=None, info=attempt_info)
        for attempt in range(1, max_attempts_per_sample + 1):
            attempt_info["attempts"] = attempt
            candidate = perturb_structure(seed, config, rng)
            if candidate.accepted:
                merged_info = {**attempt_info, **candidate.info}
                result = PerturbationResult(atoms=candidate.atoms, info=merged_info)
                break
            attempt_info[f"reject_{attempt}"] = candidate.info
        results.append(result)
    return results


def apply_hookean_repair(atoms: Atoms, config: BoxSamplingConfig) -> int:
    """Push apart pairs of atoms closer than the configured minimum.

    Iterates: for every pair whose distance is below
    ``config.min_pair_distance_ang[pair]``, displaces both atoms along the
    bond axis so the new distance equals the cutoff. Loops until no pair
    violates the constraint or ``config.hookean_max_steps`` is reached.

    Not a real geometry optimization — DFT will do that. Purpose is to
    prevent the SCF from diverging on stacked atoms.

    Args:
        atoms: Structure to repair *in place*.
        config: Box-sampling configuration (provides minimum distances and
            step budget).

    Returns:
        Number of iterations actually executed (0 if no violations on entry).
    """
    for step in range(config.hookean_max_steps):
        violation = _worst_pair_violation(atoms, config.min_pair_distance_ang)
        if violation is None:
            return step
        i, j, d, r_min = violation
        _push_apart(atoms, i, j, d, r_min)
    return config.hookean_max_steps


def enforce_cu_o_connectivity(atoms: Atoms, cutoff_ang: float) -> bool:
    """Check that every O atom has at least one Cu within ``cutoff_ang``.

    A failing O means the perturbation produced a floating oxygen that
    isn't bonded to the Cu lattice — non-physical for the Cu-O dataset, and
    a known way for box-sampling to waste DFT cycles.

    Args:
        atoms: Structure to check.
        cutoff_ang: Maximum Cu-O distance considered a bond (Å).

    Returns:
        True if all O atoms are within ``cutoff_ang`` of at least one Cu;
        True trivially if there are no O atoms; False otherwise.
    """
    o_indices = [i for i, sym in enumerate(atoms.get_chemical_symbols()) if sym == "O"]
    cu_indices = [i for i, sym in enumerate(atoms.get_chemical_symbols()) if sym == "Cu"]
    if not o_indices:
        return True
    if not cu_indices:
        return False

    distances = atoms.get_all_distances(mic=True)
    return all(np.any(distances[o, cu_indices] <= cutoff_ang) for o in o_indices)


def _delete_random_oxygens(atoms: Atoms, n: int, rng: np.random.Generator) -> int:
    """Delete up to ``n`` random O atoms in place. Returns how many were removed."""
    if n <= 0:
        return 0
    o_indices = [i for i, sym in enumerate(atoms.get_chemical_symbols()) if sym == "O"]
    if not o_indices:
        return 0
    n_actual = min(n, len(o_indices))
    chosen = rng.choice(len(o_indices), size=n_actual, replace=False)
    to_delete = sorted([o_indices[k] for k in chosen], reverse=True)
    for idx in to_delete:
        del atoms[idx]
    return n_actual


def _insert_random_oxygens(
    atoms: Atoms,
    n: int,
    config: BoxSamplingConfig,
    rng: np.random.Generator,
) -> int:
    """Insert up to ``n`` random O atoms in place. Returns how many succeeded."""
    if n <= 0:
        return 0
    inserted = 0
    for _ in range(n):
        if _try_insert_one_oxygen(atoms, config, rng):
            inserted += 1
    return inserted


def _try_insert_one_oxygen(
    atoms: Atoms,
    config: BoxSamplingConfig,
    rng: np.random.Generator,
) -> bool:
    """Try to insert one O atom at a random position. Return True on success."""
    cell = np.asarray(atoms.cell.array)
    for _ in range(config.insert_attempts):
        frac = rng.uniform(0.0, 1.0, size=3)
        candidate_pos = frac @ cell
        if _min_distance_to_existing(atoms, candidate_pos) >= config.insert_min_distance_ang:
            atoms.append(Atom("O", position=candidate_pos))
            return True
    return False


def _min_distance_to_existing(atoms: Atoms, position: np.ndarray) -> float:
    """Minimum distance from ``position`` to any atom (with periodic images)."""
    if len(atoms) == 0:
        return np.inf
    # Use a temporary atoms object to leverage ASE's MIC distance machinery.
    probe = atoms.copy()
    probe.append(Atom("X", position=position))
    distances = probe.get_distances(len(probe) - 1, list(range(len(atoms))), mic=True)
    return float(np.min(distances))


def _worst_pair_violation(
    atoms: Atoms,
    min_pair_distance: dict[frozenset[str], float],
) -> tuple[int, int, float, float] | None:
    """Return the closest violating (i, j, d, r_min) or None if all pairs OK."""
    n = len(atoms)
    if n < 2:
        return None
    symbols = atoms.get_chemical_symbols()
    distances = atoms.get_all_distances(mic=True)

    worst: tuple[int, int, float, float] | None = None
    worst_ratio = 1.0
    for i in range(n):
        for j in range(i + 1, n):
            r_min = min_pair_distance.get(frozenset([symbols[i], symbols[j]]))
            if r_min is None:
                continue
            d = float(distances[i, j])
            if d < r_min:
                ratio = d / r_min
                if ratio < worst_ratio:
                    worst_ratio = ratio
                    worst = (i, j, d, r_min)
    return worst


def _push_apart(atoms: Atoms, i: int, j: int, current_distance: float, target: float) -> None:
    """Displace atoms i and j along their bond axis to reach ``target`` separation.

    Uses the minimum-image vector so the repair respects periodic boundaries.
    Both atoms move by half of the required displacement each, so the
    centre of mass of the pair is preserved.
    """
    if current_distance <= 0.0:
        # Degenerate: pick an arbitrary axis to avoid divide-by-zero.
        direction = np.array([1.0, 0.0, 0.0])
    else:
        # Minimum-image displacement from i to j.
        vec = atoms.get_distance(i, j, mic=True, vector=True)
        direction = np.asarray(vec) / current_distance

    displacement = 0.5 * (target - current_distance)
    atoms.positions[i] -= direction * displacement
    atoms.positions[j] += direction * displacement
