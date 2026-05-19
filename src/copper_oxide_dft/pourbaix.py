"""Pourbaix-style stability diagram construction and plotting.

A Pourbaix diagram, in this project's first cut, is the answer to the
question: "of these solid Cu-containing phases, which has the lowest per-Cu
free energy at a given (U, pH)?" The boundaries between regions are the
electrochemical equilibrium lines between adjacent phases.

This module covers the *solid-phase* Pourbaix (Cu / Cu2O / CuO and any other
Cu-containing solids the caller adds). It does NOT include dissolution to
soluble Cu²⁺ / HCuO₂⁻ species — those are real and important regions of the
experimental diagram but require additional inputs (aqueous ion energetics,
chosen ion activity); a later phase will add them. Until then, a "Cu metal"
region at very low U / acidic pH is a placeholder for the experimental
"Cu²⁺(aq)" region.

Inputs come from :mod:`copper_oxide_dft.che`:

* :class:`~copper_oxide_dft.che.PhaseEnergetics` per phase (Cu, Cu2O, CuO).
* :class:`~copper_oxide_dft.che.ReferenceEnergetics` for H2 / H2O.

Outputs are a :class:`PourbaixDiagram` with the (U, pH) grid, per-phase
ΔG_per_Cu arrays, and the argmin-phase index at each grid point. The
plotting function then renders that to matplotlib.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from copper_oxide_dft.che import (
    DEFAULT_TEMPERATURE_K,
    AdsorbateState,
    PhaseEnergetics,
    ReferenceEnergetics,
    adsorbate_state_relative_free_energy_ev,
    phase_free_energy_per_cu_ev,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes


@dataclass(frozen=True)
class PourbaixDiagram:
    """Result of a Pourbaix grid sweep.

    Attributes:
        u_grid_v: 1D array of potentials in V vs. SHE.
        ph_grid: 1D array of pH values.
        phase_names: Names of the phases, in the order they were passed in
            (matches the species axis of ``free_energies_per_cu_ev``).
        free_energies_per_cu_ev: Array of shape
            ``(len(phase_names), len(ph_grid), len(u_grid_v))``
            of per-Cu free energies. Indexing is ``[phase, i_ph, i_u]``.
        stable_phase_index: Array of shape ``(len(ph_grid), len(u_grid_v))``
            giving the argmin-phase index at each (pH, U) grid point.
    """

    u_grid_v: np.ndarray
    ph_grid: np.ndarray
    phase_names: tuple[str, ...]
    free_energies_per_cu_ev: np.ndarray
    stable_phase_index: np.ndarray

    def stable_phase_at(self, u_she_v: float, ph: float) -> str:
        """Name of the stable phase at the grid point nearest (U, pH).

        For an off-grid point this returns the nearest-neighbor value
        rather than interpolating; that's sufficient when the grid is
        fine compared to the boundary slope (~59 mV/pH-unit).
        """
        i_u = int(np.argmin(np.abs(self.u_grid_v - u_she_v)))
        i_ph = int(np.argmin(np.abs(self.ph_grid - ph)))
        return self.phase_names[int(self.stable_phase_index[i_ph, i_u])]


def phase_diagram(
    phases: Sequence[PhaseEnergetics],
    cu_metal_reference: PhaseEnergetics,
    water_reference: ReferenceEnergetics,
    *,
    u_range_v: tuple[float, float] = (-1.0, 1.0),
    u_steps: int = 81,
    ph_range: tuple[float, float] = (0.0, 14.0),
    ph_steps: int = 71,
    temperature_k: float = DEFAULT_TEMPERATURE_K,
) -> PourbaixDiagram:
    """Build a Pourbaix-style (U, pH) stability map for the given phases.

    For each (U, pH) grid point, the per-Cu free energy of every phase is
    computed via :func:`~copper_oxide_dft.che.phase_free_energy_per_cu_ev`,
    and the lowest-energy phase is recorded as the stable one.

    Args:
        phases: Solid phases to compete. Must each have ``n_cu>0``.
            The Cu-metal reference may be included here too — it will
            give ΔG_per_Cu = 0 across the diagram and serve as the "Cu"
            stability region.
        cu_metal_reference: Per-Cu energy zero. Typically the same
            ``PhaseEnergetics`` object that appears in ``phases``.
        water_reference: H2 / H2O reference for μ(O)(U, pH).
        u_range_v: (min, max) potentials in V vs. SHE.
        u_steps: Number of grid points along U (inclusive endpoints).
        ph_range: (min, max) pH values.
        ph_steps: Number of grid points along pH.
        temperature_k: Temperature in K.

    Returns:
        :class:`PourbaixDiagram` with grid axes, per-phase free-energy
        arrays, and the stable-phase-index map.

    Raises:
        ValueError: If ``phases`` is empty, or if a grid size is < 2.
    """
    if not phases:
        raise ValueError("Pass at least one phase to phase_diagram().")
    if u_steps < 2 or ph_steps < 2:
        raise ValueError("u_steps and ph_steps must each be >= 2.")

    u_grid = np.linspace(u_range_v[0], u_range_v[1], u_steps)
    ph_grid = np.linspace(ph_range[0], ph_range[1], ph_steps)
    free_energies = np.empty((len(phases), len(ph_grid), len(u_grid)))

    for i_phase, phase in enumerate(phases):
        for i_ph, ph in enumerate(ph_grid):
            for i_u, u in enumerate(u_grid):
                free_energies[i_phase, i_ph, i_u] = phase_free_energy_per_cu_ev(
                    phase,
                    cu_metal_reference,
                    water_reference,
                    u_she_v=float(u),
                    ph=float(ph),
                    temperature_k=temperature_k,
                )

    stable_index = np.argmin(free_energies, axis=0)
    return PourbaixDiagram(
        u_grid_v=u_grid,
        ph_grid=ph_grid,
        phase_names=tuple(phase.name for phase in phases),
        free_energies_per_cu_ev=free_energies,
        stable_phase_index=stable_index,
    )


def adsorbate_phase_diagram(
    states: Sequence[AdsorbateState],
    clean_slab: AdsorbateState,
    water_reference: ReferenceEnergetics,
    *,
    u_range_v: tuple[float, float] = (-1.0, 1.0),
    u_steps: int = 81,
    ph_range: tuple[float, float] = (0.0, 14.0),
    ph_steps: int = 71,
    temperature_k: float = DEFAULT_TEMPERATURE_K,
) -> PourbaixDiagram:
    """Phase-4-v2 surface Pourbaix for O/OH coverages on Cu(111).

    For each (U, pH) grid point, compute ΔG of every coverage state
    relative to the clean slab, then pick the lowest. Reuses the
    :class:`PourbaixDiagram` shape from the bulk diagram so existing
    plotting code works unchanged.

    Args:
        states: Adsorbate states to compete. Must all share the same
            underlying supercell.
        clean_slab: Bare Cu(111) reference; should also appear in
            ``states`` so the diagram has a "clean" region.
        water_reference: H2 / H2O DFT energetics.
        u_range_v: (min, max) potentials (V vs. SHE).
        u_steps: U grid points.
        ph_range: (min, max) pH.
        ph_steps: pH grid points.
        temperature_k: Temperature (K).

    Returns:
        :class:`PourbaixDiagram` whose ``free_energies_per_cu_ev`` field
        holds *absolute* ΔG values (not per-Cu — the per-Cu field name
        is reused for type compatibility; the values are eV, not eV/Cu).

    Raises:
        ValueError: If ``states`` is empty or a grid size is < 2.
    """
    if not states:
        raise ValueError(
            "Pass at least one adsorbate state to adsorbate_phase_diagram()."
        )
    if u_steps < 2 or ph_steps < 2:
        raise ValueError("u_steps and ph_steps must each be >= 2.")

    u_grid = np.linspace(u_range_v[0], u_range_v[1], u_steps)
    ph_grid = np.linspace(ph_range[0], ph_range[1], ph_steps)
    free_energies = np.empty((len(states), len(ph_grid), len(u_grid)))

    for i_state, state in enumerate(states):
        for i_ph, ph in enumerate(ph_grid):
            for i_u, u in enumerate(u_grid):
                free_energies[i_state, i_ph, i_u] = (
                    adsorbate_state_relative_free_energy_ev(
                        state,
                        clean_slab,
                        water_reference,
                        u_she_v=float(u),
                        ph=float(ph),
                        temperature_k=temperature_k,
                    )
                )

    stable_index = np.argmin(free_energies, axis=0)
    return PourbaixDiagram(
        u_grid_v=u_grid,
        ph_grid=ph_grid,
        phase_names=tuple(s.name for s in states),
        free_energies_per_cu_ev=free_energies,
        stable_phase_index=stable_index,
    )


def plot_diagram(
    diagram: PourbaixDiagram,
    *,
    ax: Axes | None = None,
    mark_point: tuple[float, float] | None = None,
    title: str | None = None,
    cmap: str = "Set3",
) -> Axes:
    """Render a Pourbaix diagram with matplotlib.

    Each phase region is shown as a flat color; if ``mark_point`` is given,
    a marker is drawn at that (U, pH) and the stable phase there is added
    to the title.

    Args:
        diagram: Diagram produced by :func:`phase_diagram`.
        ax: Existing Axes to draw into; if None, a new figure+axes is
            created.
        mark_point: (U, pH) to overlay as a marker; typically the
            experimental condition of interest.
        title: Custom plot title. Defaults to a description of the
            stable phase at ``mark_point`` if given.
        cmap: Matplotlib qualitative colormap. ``Set3`` works well up to
            ~10 phases.

    Returns:
        The matplotlib :class:`~matplotlib.axes.Axes` containing the plot.
    """
    import matplotlib.pyplot as plt
    from matplotlib import colors as mpl_colors

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    n_phases = len(diagram.phase_names)
    # Discrete colormap with one entry per phase, so a flat phase region
    # comes out as a single solid color.
    cmap_obj = plt.get_cmap(cmap, n_phases)
    norm = mpl_colors.BoundaryNorm(np.arange(-0.5, n_phases + 0.5, 1.0), cmap_obj.N)

    ax.imshow(
        diagram.stable_phase_index,
        origin="lower",
        extent=(
            diagram.u_grid_v[0],
            diagram.u_grid_v[-1],
            diagram.ph_grid[0],
            diagram.ph_grid[-1],
        ),
        aspect="auto",
        cmap=cmap_obj,
        norm=norm,
        interpolation="nearest",
    )
    ax.set_xlabel("U vs. SHE (V)")
    ax.set_ylabel("pH")

    # Custom legend with one patch per phase.
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=cmap_obj(i), edgecolor="black", label=name)
        for i, name in enumerate(diagram.phase_names)
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=False,
        title="Stable phase",
    )

    if mark_point is not None:
        u_m, ph_m = mark_point
        ax.plot(u_m, ph_m, marker="x", markersize=12, mew=2.5, color="black")
        stable = diagram.stable_phase_at(u_m, ph_m)
        ax.annotate(
            f"({u_m:+.2f} V, pH {ph_m})\nstable: {stable}",
            xy=(u_m, ph_m),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
        )

    if title is None and mark_point is not None:
        stable = diagram.stable_phase_at(*mark_point)
        title = f"Cu-O Pourbaix (CHE): stable at marked point = {stable}"
    if title is not None:
        ax.set_title(title)

    return ax
