"""Grand-canonical genetic algorithm wrapper over GOCIA.

Drives a structural search on a Cu(111) substrate using a fine-tuned
MACE potential as the energy/force backend. Mirrors
:doc:`/docs/machine-learned-dft.md` §4:

- The fitness is the grand potential ``Ω_O = E - μ_O · N_O``.
- Mutations rattle, insert / delete O on the active region of the
  substrate.
- An *unbiased* sweep over ``μ_O ∈ [-7.0, -6.0] eV`` finds the natural
  ground state at each chemical potential.
- A *biased* sweep adds a sum of Gaussian potentials on the
  stoichiometry ``x_O`` to fill in the metastable intermediate-coverage
  states the unbiased sweep skips over.

The pure math (grand potential, Gaussian bias, x_O computation) is
tested here. The GCGA loop itself lazy-imports :mod:`gocia` and is
exercised only on the DGX Spark workstation — the wrapper is
deliberately thin so the GOCIA API surface (which has reshuffled across
releases) can be re-pinned in one place without touching the rest of
the pipeline.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from ase import Atoms

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


@dataclass(frozen=True)
class GCGAConfig:
    """All parameters for one GCGA search.

    Attributes:
        substrate: Cu(111) slab Atoms to search over.
        active_indices: Indices of atoms GCGA may displace or delete.
            The complement is treated as fixed substrate.
        active_species: Chemical symbols GCGA may insert. Default
            ``("O",)`` matches the pivot decision (no Cu mobility in
            this round).
        mu_o_ev: Grand-canonical O chemical potential (eV vs vacuum)
            for the fitness function.
        n_generations: GA outer loop length.
        population_size: GA population at each generation.
        mutation_rate: Probability per offspring of an O insert/delete.
        rattle_stdev_ang: Per-atom rattle for crossover/mutation.
        bias_centers: x_O values at which to add Gaussian bias bumps.
            Empty for the unbiased pass.
        bias_amplitude_ev: Height of each Gaussian (eV).
        bias_sigma: Width of each Gaussian (in x_O units).
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
    mutation_rate: float = 0.3
    rattle_stdev_ang: float = 0.1
    bias_centers: tuple[float, ...] = ()
    bias_amplitude_ev: float = DEFAULT_BIASED_AMPLITUDE_EV
    bias_sigma: float = DEFAULT_BIASED_SIGMA
    rng_seed: int = 0
    max_atoms: int = 500


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
        O insertions land.

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


# ---------- GCGA runner (lazy GOCIA import) ----------------------------------


def run_gcga_sweep(
    config: GCGAConfig,
    *,
    mace_model_path: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    device: str = "cuda",
):
    """Drive one GOCIA GCGA at a fixed μ_O (and optional bias).

    This is the only function in the module that touches :mod:`gocia`.
    On first run on DGX Spark, the GOCIA API surface must be pinned —
    see :doc:`/docs/ml-gcgo-pivot.md` §7 risk #1.

    Outline of the GOCIA call (subject to API verification):

    1. Build a :class:`gocia.interface.Interface` from
       ``config.substrate`` and ``config.active_indices``.
    2. Attach a MACE calculator factory that wraps each candidate's
       potential energy via :func:`biased_grand_potential_ev`.
    3. Run :func:`gocia.popGen.evolve` (or the current equivalent) for
       ``config.n_generations`` × ``config.population_size`` candidates.
    4. Persist the final population as ``out_dir/population.json`` and
       the full trajectory as ``out_dir/trajectory.xyz``.

    Args:
        config: GCGA configuration (substrate, μ_O, bias, ...).
        mace_model_path: Fine-tuned MACE model.
        out_dir: Directory to write the resulting population.
        device: ``"cuda"`` on DGX Spark, ``"cpu"`` for ad-hoc tests.

    Raises:
        ImportError: If GOCIA / MACE are not installed.
        NotImplementedError: Until the GOCIA API is pinned. This is
            deliberate so the function fails loudly rather than silently
            running a wrong API call.
    """
    # The GOCIA wiring lives here. Pin once on first DGX Spark run.
    # Raise before any side effects (filesystem writes, heavy imports) so a
    # stubbed run doesn't leave detritus behind.
    raise NotImplementedError(
        "GOCIA invocation not yet pinned to a specific API version. "
        "On first run on DGX Spark, replace this NotImplementedError "
        "with the actual gocia.popGen.evolve(...) call against the "
        "installed GOCIA release. The math (biased_grand_potential_ev) "
        "and the substrate (build_cu111_gcga_substrate) are already correct. "
        "Both `gocia` and `mace.calculators` will be lazy-imported at "
        "that point — they are not imported now to avoid masking the "
        "stub behind an unrelated ImportError."
    )
