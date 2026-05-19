"""Grand-canonical genetic algorithm on Cu(111) using ase-ga.

Drives a structural search on a Cu(111) substrate using a fine-tuned
MACE potential as the energy/force backend. Mirrors
:doc:`/docs/machine-learned-dft.md` §4:

- The fitness is the grand potential ``Ω_O = E - μ_O · N_O``.
- Mutations rattle active atoms, insert O above the active region, or
  delete an active O.
- An *unbiased* sweep over ``μ_O ∈ [-7.0, -6.0] eV`` finds the natural
  ground state at each chemical potential.
- A *biased* sweep adds a sum of Gaussian potentials on the
  stoichiometry ``x_O`` to fill in the metastable intermediate-coverage
  states the unbiased sweep skips over.

The GA backend is **ase-ga** (the spin-out of ``ase.ga`` maintained by
the DTU/CAMD ASE core team — see
[dtu-energy/ase-ga](https://github.com/dtu-energy/ase-ga)). The
project switched from GOCIA to ase-ga on 2026-05-19 for community
maintenance reasons; see :doc:`/docs/ground_truths.md` 2026-05-19 entry
and :doc:`/docs/ml-gcgo-pivot.md`. The grand-canonical math
(:func:`grand_potential_ev`, :func:`gaussian_bias_ev`,
:func:`biased_grand_potential_ev`, :func:`compute_x_o`) is unchanged
across that pivot.

Atom-ordering convention (load-bearing): the substrate produced by
:func:`build_cu111_gcga_substrate` places the **active atoms last** in
the ``Atoms`` list, matching ase-ga's slab/top convention
(``slab = atoms[:n_slab]; top = atoms[n_slab:]``). The mutation
operators in this module assume that convention; the
``run_gcga_sweep`` driver validates it on entry.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ase import Atom, Atoms

DEFAULT_MU_O_RANGE_EV: tuple[float, float] = (-7.0, -6.0)
"""Manuscript μ_O sweep range for the unbiased GCGA pass."""

DEFAULT_MU_O_N_POINTS = 11
"""How many μ_O values to sweep across the unbiased range."""

DEFAULT_BIASED_X_O_RANGE: tuple[float, float] = (0.32, 1.0)
"""x_O range to fill in via Gaussian-biased GCGA. Manuscript values."""

DEFAULT_BIASED_AMPLITUDE_EV = 0.5
"""Height of each Gaussian bias bump (eV). Trades off "how strongly the
algorithm is pushed away from the local minimum" against "how much we
distort the ranking we care about." 0.5 eV is a defensible starting
point; verify in practice that biased and unbiased ensembles agree on
the structures both pick up."""

DEFAULT_BIASED_SIGMA = 0.05
"""Width of each Gaussian (in x_O units). 0.05 ≈ one of ~20 x_O bins."""

DEFAULT_LAYERS_GCGA = 12
"""Cu(111) layer count. Manuscript value — top 6 are active for O
manipulation; bottom 6 act as semi-infinite substrate."""

DEFAULT_ACTIVE_TOP_LAYERS = 6
"""How many top layers GCGA may modify."""

DEFAULT_LATERAL_GCGA: tuple[int, int] = (4, 4)
"""Lateral repetition of the (1×1) Cu(111) primitive cell. Starting
point per :doc:`/docs/ml-gcgo-pivot.md` §3.5; revisit after the first
ensemble."""

DEFAULT_MIN_PAIR_DISTANCE_ANG: dict[frozenset[str], float] = {
    frozenset(["Cu", "Cu"]): 1.8,
    frozenset(["Cu", "O"]): 1.4,
    frozenset(["O", "O"]): 1.0,
}
"""Pair-specific minimum distances (Å) for the rattle / insert filters.
Same values as the box-sampling stage so the GCGA can never produce a
structure tighter than what we sent to QE during training."""

DEFAULT_INSERT_ATTEMPTS = 50
"""Random-position tries per O insertion before giving up."""

DEFAULT_INSERT_Z_PADDING_ANG = 1.5
"""How far above the current top active atom an inserted O is allowed
to land (Å). Combined with the slab top z as the floor, this is the
vertical band where insertions are sampled."""

DEFAULT_OPERATOR_WEIGHTS: tuple[float, float, float] = (0.5, 0.25, 0.25)
"""Probability of each operator per offspring: (rattle, insert, remove).
Manuscript-compatible default — rattle dominates so the GA stays inside
the chemical neighbourhood of the current best while still exploring
composition."""

DEFAULT_TOURNAMENT_K = 3
"""k for k-way tournament selection. 2-3 is standard; larger k makes the
search more greedy (faster convergence, more local-minimum traps)."""

DEFAULT_MAX_ATOMS = 500
"""Upper bound on cell size; samples that grow past this via insertions
are rejected. Protects against runaway insertions that would OOM the
MACE evaluator."""


@dataclass(frozen=True)
class GCGAConfig:
    """All parameters for one GCGA search.

    Attributes:
        substrate: Cu(111) slab Atoms to search over. Active atoms must
            be the contiguous **tail** of the ``Atoms`` list (ase-ga
            slab/top convention).
        active_indices: Indices of atoms GCGA may displace or delete.
            Must equal ``range(len(substrate)-len(active_indices), len(substrate))``.
        active_species: Chemical symbols GCGA may insert. Default
            ``("O",)`` matches the pivot decision (no Cu mobility in
            this round).
        mu_o_ev: Grand-canonical O chemical potential (eV vs vacuum)
            for the fitness function.
        n_generations: GA outer loop length.
        population_size: GA population at each generation.
        operator_weights: ``(rattle, insert, remove)`` probabilities,
            normalised before sampling.
        rattle_stdev_ang: Per-atom rattle strength (Å). Passed to
            ase-ga's ``RattleMutation`` as ``rattle_strength`` (which
            samples each component uniformly from ``[-strength, strength]``).
        bias_centers: x_O values at which to add Gaussian bias bumps.
            Empty for the unbiased pass.
        bias_amplitude_ev: Height of each Gaussian (eV).
        bias_sigma: Width of each Gaussian (in x_O units).
        min_pair_distance_ang: Pair-specific minimum distances (Å) below
            which a candidate is rejected.
        insert_attempts: Random-position tries per insertion.
        insert_z_padding_ang: Vertical headroom above the current top
            active atom for new insertions (Å).
        tournament_k: k-way tournament selection size.
        rng_seed: GA RNG seed for reproducibility.
        max_atoms: Upper bound on cell size; samples that grow past
            this are rejected.
    """

    substrate: Atoms
    active_indices: tuple[int, ...]
    mu_o_ev: float
    active_species: tuple[str, ...] = ("O",)
    n_generations: int = 50
    population_size: int = 50
    operator_weights: tuple[float, float, float] = DEFAULT_OPERATOR_WEIGHTS
    rattle_stdev_ang: float = 0.1
    bias_centers: tuple[float, ...] = ()
    bias_amplitude_ev: float = DEFAULT_BIASED_AMPLITUDE_EV
    bias_sigma: float = DEFAULT_BIASED_SIGMA
    min_pair_distance_ang: dict[frozenset[str], float] = field(
        default_factory=lambda: dict(DEFAULT_MIN_PAIR_DISTANCE_ANG)
    )
    insert_attempts: int = DEFAULT_INSERT_ATTEMPTS
    insert_z_padding_ang: float = DEFAULT_INSERT_Z_PADDING_ANG
    tournament_k: int = DEFAULT_TOURNAMENT_K
    rng_seed: int = 0
    max_atoms: int = DEFAULT_MAX_ATOMS


# ---------- Pure math: grand potential, Gaussian bias, x_O -------------------


def compute_x_o(atoms: Atoms) -> float:
    """Stoichiometry x_O = N_O / (N_Cu + N_O). Returns 0.0 for pure Cu."""
    n_cu = sum(1 for s in atoms.get_chemical_symbols() if s == "Cu")
    n_o = sum(1 for s in atoms.get_chemical_symbols() if s == "O")
    total = n_cu + n_o
    if total == 0:
        return 0.0
    return float(n_o) / float(total)


def grand_potential_ev(
    energy_ev: float, atoms: Atoms, mu_o_ev: float
) -> float:
    """Ω_O = E - μ_O · N_O. Manuscript :doc:`/docs/machine-learned-dft.md` §4.

    Args:
        energy_ev: Total DFT/MACE energy of the configuration (eV).
        atoms: Configuration (used only for counting O atoms).
        mu_o_ev: Oxygen chemical potential (eV vs the reference used to
            compute ``energy_ev`` — vacuum in our case).

    Returns:
        Grand potential in eV.
    """
    n_o = sum(1 for s in atoms.get_chemical_symbols() if s == "O")
    return float(energy_ev) - mu_o_ev * float(n_o)


def gaussian_bias_ev(
    x_o: float,
    centers: Sequence[float],
    amplitude_ev: float,
    sigma: float,
) -> float:
    """Sum of Gaussian penalties used in the biased GCGA pass.

    ``bias(x_O) = Σ_k A · exp(-(x_O - c_k)² / (2 σ²))``

    Adding this to the grand potential pushes the algorithm away from
    local minima and toward sampling the chosen ``centers``. Manuscript
    §4 step 3.

    Args:
        x_o: Current configuration's stoichiometry.
        centers: x_O values to place Gaussians at.
        amplitude_ev: Height of each Gaussian (eV).
        sigma: Width of each Gaussian (dimensionless, x_O units).

    Returns:
        Total bias in eV.

    Raises:
        ValueError: If ``sigma <= 0``.
    """
    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive; got {sigma}.")
    if not centers:
        return 0.0
    x = float(x_o)
    bumps = np.exp(-((x - np.asarray(centers, dtype=float)) ** 2) / (2.0 * sigma ** 2))
    return float(amplitude_ev * bumps.sum())


def biased_grand_potential_ev(
    energy_ev: float,
    atoms: Atoms,
    config: GCGAConfig,
) -> float:
    """Grand potential plus configured Gaussian bias.

    Convenience: ``grand_potential_ev(...) + gaussian_bias_ev(x_o, ...)``.
    """
    omega = grand_potential_ev(energy_ev, atoms, config.mu_o_ev)
    bias = gaussian_bias_ev(
        compute_x_o(atoms),
        config.bias_centers,
        config.bias_amplitude_ev,
        config.bias_sigma,
    )
    return omega + bias


# ---------- Substrate construction --------------------------------------------


def build_cu111_gcga_substrate(
    *,
    layers: int = DEFAULT_LAYERS_GCGA,
    lateral: tuple[int, int] = DEFAULT_LATERAL_GCGA,
    active_top_layers: int = DEFAULT_ACTIVE_TOP_LAYERS,
    lattice_a_ang: float | None = None,
    vacuum_ang: float = 20.0,
) -> tuple[Atoms, tuple[int, ...]]:
    """Build the 12-layer Cu(111) substrate + index list of active atoms.

    Uses :func:`copper_oxide_dft.structure_builder.build_cu111_slab` so
    the constraint convention (bottom layers fixed) and the lattice
    parameter come from the project's settings.

    Args:
        layers: Total slab layers.
        lateral: Lateral repetition of the primitive cell.
        active_top_layers: How many top layers GCGA may modify.
        lattice_a_ang: PBE-relaxed Cu lattice parameter. Defaults to
            :data:`structure_builder.CU_LATTICE_PARAMETER_ANG` (experimental
            3.615 Å); production callers must pass
            ``load_config("configs/converged.json").systems["bulk_cu"].extras["lattice_a_ang"]``.
        vacuum_ang: Vacuum thickness above the slab (Å).

    Returns:
        ``(slab, active_indices)``. ``active_indices`` lists atom indices
        (into ``slab``) that fall in the top ``active_top_layers`` z-layers
        — these are the atoms GCGA is allowed to displace, and where new
        O insertions land. By construction these are the contiguous tail
        of the ``Atoms`` list (ase-ga slab/top convention).

    Raises:
        ValueError: If ``active_top_layers > layers`` or either is < 1.
    """
    if layers < 1:
        raise ValueError(f"layers must be >= 1; got {layers}.")
    if active_top_layers < 1 or active_top_layers > layers:
        raise ValueError(
            f"active_top_layers must be in [1, layers]; got {active_top_layers} "
            f"with layers={layers}."
        )

    # Lazy import — avoids a circular import surface when ml/__init__ pulls in
    # gcga at package-import time.
    from copper_oxide_dft.structure_builder import (
        CU_LATTICE_PARAMETER_ANG,
        build_cu111_slab,
        summarize_layers,
    )

    a = lattice_a_ang if lattice_a_ang is not None else CU_LATTICE_PARAMETER_ANG
    slab = build_cu111_slab(
        layers=layers,
        supercell=lateral,
        vacuum_ang=vacuum_ang,
        a=a,
        fix_bottom_layers=layers - active_top_layers,
    )

    z_layers = summarize_layers(slab)
    if len(z_layers) < layers:
        # Pathological: the slab builder produced fewer z-layers than asked
        # for (shouldn't happen in normal use, but be defensive).
        active_layer_zs = [z_layers[-i].z for i in range(1, len(z_layers) + 1)]
    else:
        active_layer_zs = [z_layers[-i].z for i in range(1, active_top_layers + 1)]
    threshold_z = min(active_layer_zs) - 1e-3
    active = tuple(i for i, atom in enumerate(slab) if atom.z >= threshold_z)
    return slab, active


# ---------- Helpers shared by the operators and the driver --------------------


def _blmin_atomic_numbers(
    min_pair_distance_ang: dict[frozenset[str], float],
) -> dict[tuple[int, int], float]:
    """Convert {frozenset(symbols): min_dist} → {(Z_a, Z_b): min_dist}.

    ase-ga's ``RattleMutation`` and ``atoms_too_close`` look up blmin by
    atomic-number tuples in either ordering. We emit both ``(a, b)`` and
    ``(b, a)`` so the lookup never misses.
    """
    from ase.data import atomic_numbers

    out: dict[tuple[int, int], float] = {}
    for pair, dist in min_pair_distance_ang.items():
        symbols = list(pair)
        # Single-element pairs come through as one-element frozensets.
        if len(symbols) == 1:
            z = atomic_numbers[symbols[0]]
            out[(z, z)] = float(dist)
            continue
        z1 = atomic_numbers[symbols[0]]
        z2 = atomic_numbers[symbols[1]]
        out[(z1, z2)] = float(dist)
        out[(z2, z1)] = float(dist)
    return out


def _validate_active_is_contiguous_tail(
    n_atoms: int, active_indices: Sequence[int]
) -> int:
    """Confirm active atoms occupy the tail of the Atoms list.

    Returns the implied ``n_slab`` (number of fixed substrate atoms).
    Raises ``ValueError`` if the convention is violated.
    """
    if len(active_indices) == 0:
        return n_atoms
    expected = tuple(range(n_atoms - len(active_indices), n_atoms))
    if tuple(sorted(active_indices)) != expected:
        raise ValueError(
            f"active_indices must be the contiguous tail of the substrate "
            f"[{expected[0]}..{expected[-1]}]; got {tuple(sorted(active_indices))}. "
            f"This is the ase-ga slab/top convention; see "
            f"docs/ground_truths.md 2026-05-19 entry."
        )
    return n_atoms - len(active_indices)


# ---------- Mutation operators (ase-ga compatible) ----------------------------


def rattle_offspring(
    parent: Atoms,
    n_slab: int,
    blmin: dict[tuple[int, int], float],
    rattle_strength: float,
    rng: np.random.Generator,
) -> Atoms | None:
    """Rattle the active (top) atoms via ase-ga's ``RattleMutation``.

    Atoms ``parent[:n_slab]`` are kept fixed; ``parent[n_slab:]`` are
    perturbed. Returns ``None`` if no rattled configuration could
    satisfy the ``blmin`` constraints within ase-ga's internal retry
    budget.

    Args:
        parent: Source structure.
        n_slab: Number of fixed substrate atoms (parent[:n_slab]).
        blmin: ase-ga ``(Z_a, Z_b) → min_dist`` table.
        rattle_strength: Per-atom rattle strength in Å (uniform, not σ).
        rng: NumPy generator (ase-ga's ``RattleMutation`` consumes it).
    """
    from ase_ga.standardmutations import RattleMutation

    n_top = len(parent) - n_slab
    if n_top <= 0:
        return None
    op = RattleMutation(
        blmin=blmin,
        n_top=n_top,
        rattle_strength=rattle_strength,
        rng=rng,
    )
    seed = _wrap_atoms_for_ase_ga(parent)
    offspring, _descriptor = op.get_new_individual([seed])
    return offspring


def insert_oxygen_offspring(
    parent: Atoms,
    n_slab: int,
    min_pair_distance_ang: dict[frozenset[str], float],
    max_attempts: int,
    z_padding_ang: float,
    rng: np.random.Generator,
) -> Atoms | None:
    """Append one O atom at a random position above the active band.

    The insertion site is sampled uniformly in lateral (x, y) and in a
    z-band running from the top of the *slab* (fixed) region up to
    ``max(active_z) + z_padding_ang``. Rejected if the candidate ends
    up closer than the per-pair minimum to any existing atom, retried
    up to ``max_attempts`` times.

    Returns ``None`` if no valid position was found, or if there is no
    slab region to anchor the z-band to.
    """
    from ase.data import atomic_numbers

    new = parent.copy()
    if n_slab > 0:
        z_floor = max(new[i].z for i in range(n_slab))
    elif len(new) > 0:
        z_floor = min(a.z for a in new)
    else:
        # Empty parent — no geometric anchor; refuse to invent one.
        return None
    active_zs = [new[i].z for i in range(n_slab, len(new))]
    z_ceiling = (max(active_zs) if active_zs else z_floor) + z_padding_ang
    if z_ceiling <= z_floor:
        z_ceiling = z_floor + z_padding_ang

    z_o = atomic_numbers["O"]
    cell = np.asarray(new.cell.array)
    # In-plane lattice vectors. fcc111 (and every other ASE slab builder)
    # places z perpendicular to the surface, so a1 and a2 have z=0; we
    # take xy components only.
    a1_xy = cell[0, :2]
    a2_xy = cell[1, :2]

    for _ in range(max_attempts):
        u, v = rng.uniform(0.0, 1.0, size=2)
        xy = u * a1_xy + v * a2_xy
        z = rng.uniform(z_floor, z_ceiling)
        position = np.array([xy[0], xy[1], z])

        if not _insert_passes_min_distance(
            new, position, z_o, min_pair_distance_ang
        ):
            continue

        new.append(Atom("O", position=position))
        return new

    return None


def _insert_passes_min_distance(
    atoms: Atoms,
    position: np.ndarray,
    new_z: int,
    min_pair_distance_ang: dict[frozenset[str], float],
) -> bool:
    """Whether placing a ``new_z`` atom at ``position`` respects blmin.

    Uses the *symbol-keyed* min-distance table (not the ase-ga atomic-
    number form) because that is what callers usually have on hand.

    Raises:
        KeyError: If the table is missing a pair that actually occurs
            in ``atoms`` + the new species. Failing loudly here surfaces
            misconfigured tables (e.g. forgetting ``{Cu,O}``) rather
            than letting the GA silently accept overlapping atoms.
    """
    from ase.data import chemical_symbols

    new_sym = chemical_symbols[new_z]
    probe = atoms.copy()
    probe.append(Atom(new_sym, position=position))
    distances = probe.get_distances(
        len(probe) - 1, list(range(len(atoms))), mic=True
    )
    for i, d in enumerate(distances):
        pair = frozenset([new_sym, atoms[i].symbol])
        if pair not in min_pair_distance_ang:
            raise KeyError(
                f"min_pair_distance_ang is missing pair {set(pair)}; "
                f"add it before running an insertion that would form this contact."
            )
        if d < min_pair_distance_ang[pair]:
            return False
    return True


def remove_oxygen_offspring(
    parent: Atoms,
    n_slab: int,
    rng: np.random.Generator,
) -> Atoms | None:
    """Remove one random active O atom. Returns None if no active O exists."""
    o_indices = [
        i for i in range(n_slab, len(parent)) if parent[i].symbol == "O"
    ]
    if not o_indices:
        return None
    idx = int(rng.choice(o_indices))
    new = parent.copy()
    del new[idx]
    return new


def _wrap_atoms_for_ase_ga(atoms: Atoms) -> Atoms:
    """Add the ase-ga ``info`` dict scaffolding to an Atoms copy.

    ase-ga's ``OffspringCreator.finalize_individual`` writes to
    ``info["key_value_pairs"]``, so any Atoms passed in as a "parent"
    needs that key (and a ``confid``) initialised.
    """
    a = atoms.copy()
    a.info.setdefault("confid", 0)
    a.info.setdefault("key_value_pairs", {"extinct": 0})
    a.info.setdefault("data", {})
    return a


# ---------- GCGA driver -------------------------------------------------------


def _tournament_select(
    omegas: Sequence[float], k: int, rng: np.random.Generator
) -> int:
    """k-way tournament — pick k random indices, return the one with min ω."""
    n = len(omegas)
    if n == 0:
        raise ValueError("Cannot select from an empty population.")
    if n == 1:
        return 0
    candidates = rng.choice(n, size=min(k, n), replace=False)
    return int(min(candidates, key=lambda i: omegas[int(i)]))


def run_gcga_sweep(
    config: GCGAConfig,
    *,
    mace_model_path: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    device: str = "cuda",
):
    """Drive one ase-ga GCGA at a fixed μ_O (and optional bias).

    Loop is a simple in-memory (μ + λ) evolutionary strategy:

    1. Seed an initial population by rattling the substrate
       ``population_size`` times.
    2. Each generation, generate ``population_size`` offspring by
       k-way-tournament selection + random choice among
       (rattle, insert-O, remove-O), then cull to the
       ``population_size`` lowest-Ω structures.
    3. Persist the final population as ``out_dir/population.extxyz`` in
       the format :func:`copper_oxide_dft.ml.ensemble.write_ensemble_extxyz`
       expects.

    The ``rattle`` operator wraps ase-ga's
    :class:`ase_ga.standardmutations.RattleMutation`. ``insert`` and
    ``remove`` are project-specific (variable-composition operators are
    not in ase-ga's standard library); they are pure functions and
    share the rattle's atom-ordering convention.

    Args:
        config: GCGA configuration (substrate, μ_O, bias, operator mix).
        mace_model_path: Fine-tuned MACE model.
        out_dir: Directory to write the resulting population.
        device: ``"cuda"`` on DGX Spark, ``"cpu"`` for ad-hoc tests.

    Returns:
        :class:`GCGAResult` carrying the surviving population and
        per-candidate energies / grand potentials.

    Raises:
        ImportError: If ``ase-ga`` or ``mace-torch`` is not installed.
        ValueError: If the active-indices/atom-ordering convention is
            violated, or if ``population_size`` is non-positive.
    """
    # Lazy imports — both packages are heavy / GPU-bound and only land
    # on DGX Spark.
    try:
        import ase_ga  # noqa: F401 — re-imported by the operators
    except ImportError as exc:
        raise ImportError(
            "ase-ga is required to run the GCGA. Install with "
            "`pip install ase-ga`, or via `pip install -e \".[ml]\"`."
        ) from exc
    try:
        from mace.calculators import MACECalculator
    except ImportError as exc:
        raise ImportError(
            "mace-torch is required to evaluate the GCGA fitness. "
            "Install via `pip install -e \".[ml]\"`."
        ) from exc

    from copper_oxide_dft.ml.ensemble import phase_from_atoms, write_ensemble_extxyz

    if config.population_size < 1:
        raise ValueError(
            f"population_size must be >= 1; got {config.population_size}."
        )
    if config.n_generations < 0:
        raise ValueError(
            f"n_generations must be non-negative; got {config.n_generations}."
        )

    n_slab = _validate_active_is_contiguous_tail(
        len(config.substrate), config.active_indices
    )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(config.rng_seed)
    blmin_z = _blmin_atomic_numbers(config.min_pair_distance_ang)
    mace = MACECalculator(model_paths=[str(mace_model_path)], device=device)

    def evaluate(atoms: Atoms) -> tuple[float, float]:
        atoms.calc = mace
        energy_ev = float(atoms.get_potential_energy())
        omega = biased_grand_potential_ev(energy_ev, atoms, config)
        return energy_ev, omega

    # Seed the initial population by rattling the substrate.
    population: list[Atoms] = []
    energies_ev: list[float] = []
    omegas_ev: list[float] = []
    for _ in range(config.population_size):
        seed = rattle_offspring(
            config.substrate, n_slab, blmin_z, config.rattle_stdev_ang, rng
        )
        if seed is None:
            seed = config.substrate.copy()
        e, omega = evaluate(seed)
        population.append(seed)
        energies_ev.append(e)
        omegas_ev.append(omega)

    op_names = ("rattle", "insert", "remove")
    op_weights = np.asarray(config.operator_weights, dtype=float)
    op_weights = op_weights / op_weights.sum()

    source_label = "biased" if config.bias_centers else "unbiased"

    for _generation in range(config.n_generations):
        for _ in range(config.population_size):
            parent_idx = _tournament_select(omegas_ev, config.tournament_k, rng)
            parent = population[parent_idx]
            op = op_names[int(rng.choice(len(op_names), p=op_weights))]

            offspring = _apply_operator(
                op, parent, n_slab, blmin_z, config, rng
            )
            if offspring is None:
                continue
            if len(offspring) > config.max_atoms:
                continue

            e, omega = evaluate(offspring)
            population.append(offspring)
            energies_ev.append(e)
            omegas_ev.append(omega)

        # Cull to the population_size lowest-Ω survivors.
        order = sorted(range(len(omegas_ev)), key=lambda i: omegas_ev[i])
        keep = order[: config.population_size]
        population = [population[i] for i in keep]
        energies_ev = [energies_ev[i] for i in keep]
        omegas_ev = [omegas_ev[i] for i in keep]

    phases = [
        phase_from_atoms(
            atoms,
            energy_ev=energies_ev[i],
            mu_o_ev=config.mu_o_ev,
            source=source_label,
            index_in_source=i,
        )
        for i, atoms in enumerate(population)
    ]
    population_path = out_path / "population.extxyz"
    write_ensemble_extxyz(phases, population_path)

    return GCGAResult(
        population=tuple(population),
        energies_ev=tuple(energies_ev),
        omegas_biased_ev=tuple(omegas_ev),
        mu_o_ev=config.mu_o_ev,
        n_generations=config.n_generations,
        population_size=config.population_size,
        population_path=population_path,
    )


def _apply_operator(
    op: str,
    parent: Atoms,
    n_slab: int,
    blmin_z: dict[tuple[int, int], float],
    config: GCGAConfig,
    rng: np.random.Generator,
) -> Atoms | None:
    """Dispatch one mutation operator. Returns the offspring or None."""
    if op == "rattle":
        return rattle_offspring(
            parent, n_slab, blmin_z, config.rattle_stdev_ang, rng
        )
    if op == "insert":
        return insert_oxygen_offspring(
            parent,
            n_slab,
            config.min_pair_distance_ang,
            config.insert_attempts,
            config.insert_z_padding_ang,
            rng,
        )
    if op == "remove":
        return remove_oxygen_offspring(parent, n_slab, rng)
    raise ValueError(f"Unknown operator: {op!r}.")


@dataclass(frozen=True)
class GCGAResult:
    """Output of one :func:`run_gcga_sweep` call.

    Attributes:
        population: Final ``population_size`` survivors, sorted by Ω
            ascending. (Tuple, not list — the surrounding dataclass is
            frozen, and the rest of the numeric fields are tuples too.
            ASE ``Atoms`` objects themselves are still mutable; treat
            the population as read-only at the consumer.)
        energies_ev: MACE total energy per survivor (eV), aligned with
            ``population``.
        omegas_biased_ev: Biased grand potential per survivor (eV).
        mu_o_ev: μ_O the sweep was run at (eV vs vacuum).
        n_generations: Generations completed.
        population_size: Population size held at each generation.
        population_path: Where ``population.extxyz`` was written.
    """

    population: tuple[Atoms, ...]
    energies_ev: tuple[float, ...]
    omegas_biased_ev: tuple[float, ...]
    mu_o_ev: float
    n_generations: int
    population_size: int
    population_path: Path
