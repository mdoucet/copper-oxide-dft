"""Scattering Length Density (SLD) for neutron-reflectometry comparison.

After the GCGA ensemble is reduced to per-x_O minima
(:func:`copper_oxide_dft.ml.ensemble.per_x_o_minima`), each candidate
structure can be converted to an SLD depth profile and compared
directly to an experimental NR measurement.

Formula (from :doc:`/docs/machine-learned-dft.md` §5):

    SLD(z) = Σ_{atoms in slice} b_i / (A · δz)

with ``A`` = lateral cell area, ``δz`` = slice thickness (manuscript
default 10 Å), and ``b_i`` = coherent neutron scattering length per
atom. The summed-b is divided by volume to get an Å⁻² density;
conventionally reported in 10⁻⁶ Å⁻² units.

The manuscript's final step normalises the simulated SLDs by the ratio
``SLD_exp(Cu) / SLD_sim(Cu)`` to absorb the PBE-lattice-overshoot
(~1.2 % on ``a`` translates to a ~3.6 % bias on the density, which is
*not* negligible compared to typical NR sensitivities). The
:func:`bulk_cu_normalization_factor` helper computes this multiplier.

All math is pure NumPy + ASE-light — no heavy ML deps required.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from ase import Atoms

NEUTRON_SCATTERING_LENGTH_FM: dict[str, float] = {
    "Cu": 7.718,
    "O": 5.803,
    "H": -3.7406,
}
"""Coherent bound neutron scattering lengths in fm (= 10⁻⁵ Å).

Sources: NIST Center for Neutron Research tabulation.
Cu and O are positive; H is negative (which is why hydrogen-bearing
species show up as SLD *dips* in NR profiles)."""

FEMTOMETERS_TO_ANGSTROMS = 1.0e-5
"""Convert b values: 1 fm = 10⁻⁵ Å. Multiplying by this gives b in Å,
which combined with V in Å³ yields SLD in Å⁻²."""

SLD_UNIT_CONVERSION_TO_E6_PER_A2 = 1.0e6
"""Conventional NR reporting unit: 10⁻⁶ Å⁻². Multiplying an Å⁻² SLD
by this constant gives the conventional value."""

DEFAULT_INTERFACIAL_SLAB_THICKNESS_ANG = 10.0
"""Manuscript default ``δz`` for the interfacial slab (Å). The summed-b
integration is over a 10 Å z-window."""

EXPERIMENTAL_BULK_CU_SLD_E6_PER_A2 = 6.535
"""Experimental SLD of metallic Cu, in 10⁻⁶ Å⁻². Derived from
``4 × b_Cu / a³`` with b_Cu = 7.718 fm and a = 3.615 Å so the closed
form here and the constant agree. NCNR tabulations report 6.50–6.55
depending on the exact (b, density) inputs used — pick a consistent
pair, not a copied number."""


def compute_sld_summed_b_over_volume(
    atoms: Atoms,
    *,
    z_range_ang: tuple[float, float] | None = None,
    lateral_area_ang2: float | None = None,
    scattering_lengths_fm: Mapping[str, float] | None = None,
) -> float:
    """Compute one scalar SLD (10⁻⁶ Å⁻²) over a z-slab of ``atoms``.

    Args:
        atoms: ASE structure (typically a 10 Å interfacial slab
            extracted from a GCGA ensemble member).
        z_range_ang: ``(z_min, z_max)`` window in Å. Atoms outside are
            excluded from the sum. ``None`` means "use all atoms" (only
            sensible for a *bulk* cell, not a slab with vacuum).
        lateral_area_ang2: In-plane area (Å²). ``None`` reads it from
            ``atoms.cell`` via the cross product of the first two
            lattice vectors (assumes the slab z-axis is the third).
        scattering_lengths_fm: Per-species ``b`` in fm. Missing species
            raise. Defaults to :data:`NEUTRON_SCATTERING_LENGTH_FM`.

    Returns:
        SLD in 10⁻⁶ Å⁻².

    Raises:
        ValueError: If ``z_range_ang`` is degenerate or no atoms fall
            within it, or a species lacks a scattering length.
    """
    sl = (
        dict(scattering_lengths_fm)
        if scattering_lengths_fm is not None
        else dict(NEUTRON_SCATTERING_LENGTH_FM)
    )

    z = atoms.positions[:, 2]
    if z_range_ang is None:
        # Use full cell range. Treat z window as the cell's z extent.
        z_min, z_max = float(z.min()), float(z.max())
        # Add an epsilon to ensure the inclusive upper edge captures the
        # top atom in a 1-atom-thick "slab".
        z_max += 1e-9
    else:
        z_min, z_max = z_range_ang
        if not z_min < z_max:
            raise ValueError(f"z_range_ang must have z_min < z_max; got {z_range_ang}.")

    in_window = (z >= z_min) & (z <= z_max)
    if not np.any(in_window):
        raise ValueError(
            f"No atoms fall in z_range_ang={z_range_ang!r}; atoms span "
            f"({float(z.min()):.2f}, {float(z.max()):.2f}) Å."
        )

    symbols = atoms.get_chemical_symbols()
    summed_b_fm = 0.0
    for i in np.where(in_window)[0]:
        sym = symbols[int(i)]
        if sym not in sl:
            raise ValueError(
                f"No scattering length tabulated for species {sym!r}. "
                f"Pass `scattering_lengths_fm` to extend the table."
            )
        summed_b_fm += sl[sym]

    if lateral_area_ang2 is None:
        lateral_area_ang2 = _cell_lateral_area_ang2(atoms)
    if lateral_area_ang2 <= 0:
        raise ValueError(
            f"lateral_area_ang2 must be positive; got {lateral_area_ang2}."
        )

    depth_ang = z_max - z_min
    if depth_ang <= 0:
        raise ValueError(f"Depth must be positive; got {depth_ang}.")

    volume_ang3 = lateral_area_ang2 * depth_ang
    summed_b_ang = summed_b_fm * FEMTOMETERS_TO_ANGSTROMS
    sld_per_a2 = summed_b_ang / volume_ang3
    return float(sld_per_a2 * SLD_UNIT_CONVERSION_TO_E6_PER_A2)


def compute_bulk_cu_sld_e6_per_a2(
    a_ang: float,
    *,
    scattering_lengths_fm: Mapping[str, float] | None = None,
) -> float:
    """SLD of fcc Cu from its lattice parameter, in 10⁻⁶ Å⁻².

    Closed-form (no atoms object needed): ``ρ = 4 b_Cu / a³``. Used as
    the "simulated bulk Cu" reference for the manuscript's
    normalization step.

    Args:
        a_ang: Cubic fcc Cu lattice parameter (Å).
        scattering_lengths_fm: Override the default :data:`NEUTRON_SCATTERING_LENGTH_FM`.

    Returns:
        SLD in 10⁻⁶ Å⁻².
    """
    sl = scattering_lengths_fm or NEUTRON_SCATTERING_LENGTH_FM
    b_cu_ang = sl["Cu"] * FEMTOMETERS_TO_ANGSTROMS
    n_per_cell = 4  # conventional fcc cell
    volume_ang3 = a_ang**3
    return float(n_per_cell * b_cu_ang / volume_ang3 * SLD_UNIT_CONVERSION_TO_E6_PER_A2)


def bulk_cu_normalization_factor(
    *,
    simulated_a_ang: float,
    experimental_sld_e6_per_a2: float = EXPERIMENTAL_BULK_CU_SLD_E6_PER_A2,
) -> float:
    """Multiplier that converts DFT-lattice SLD to experimental scale.

    Manuscript :doc:`/docs/machine-learned-dft.md` §5.3: divide
    experimental SLD by simulated bulk-Cu SLD. The result is applied to
    every computed SLD in the dataset.

    Args:
        simulated_a_ang: The Cu lattice parameter the DFT runs were
            built on (typically the PBE-relaxed value from
            ``configs/converged.json:bulk_cu.lattice_a_ang``).
        experimental_sld_e6_per_a2: Experimental bulk-Cu SLD. Default
            value derived from a = 3.615 Å.

    Returns:
        Dimensionless factor. Typically 0.95–1.00 for PBE Cu.
    """
    simulated = compute_bulk_cu_sld_e6_per_a2(simulated_a_ang)
    if simulated == 0:
        raise ValueError("Simulated bulk-Cu SLD computed to 0 — bad lattice parameter?")
    return experimental_sld_e6_per_a2 / simulated


@dataclass(frozen=True)
class SldProfile:
    """SLD as a function of depth along z.

    ``z_centres_ang[k]`` is the centre of bin k; ``sld_e6_per_a2[k]`` is
    the SLD over that bin. Convenient for plotting against an
    experimental NR profile, which is reported in the same units.
    """

    z_centres_ang: np.ndarray
    sld_e6_per_a2: np.ndarray
    bin_width_ang: float

    def __len__(self) -> int:
        return int(self.z_centres_ang.shape[0])


def compute_sld_profile(
    atoms: Atoms,
    *,
    z_range_ang: tuple[float, float] | None = None,
    bin_width_ang: float = 1.0,
    lateral_area_ang2: float | None = None,
    scattering_lengths_fm: Mapping[str, float] | None = None,
) -> SldProfile:
    """SLD vs depth for an arbitrary slab.

    Bins atoms by z into ``bin_width_ang``-wide slices over
    ``z_range_ang``; for each bin, sums b values and divides by
    (lateral_area × bin_width). Returns the per-bin SLD as a profile.

    Args:
        atoms: ASE structure.
        z_range_ang: ``(z_min, z_max)`` window. ``None`` uses the
            atoms' actual z extent.
        bin_width_ang: Bin width in Å. Manuscript uses 10 Å for the
            interfacial-slab scalar; 1 Å gives a finer depth profile
            useful for direct NR overlay.
        lateral_area_ang2: In-plane area (Å²). Falls back to the cell.
        scattering_lengths_fm: Per-species ``b`` in fm. Defaults to
            :data:`NEUTRON_SCATTERING_LENGTH_FM`.

    Returns:
        :class:`SldProfile`.

    Raises:
        ValueError: If ``bin_width_ang <= 0`` or the z range is empty.
    """
    if bin_width_ang <= 0:
        raise ValueError(f"bin_width_ang must be positive; got {bin_width_ang}.")

    z = atoms.positions[:, 2]
    if z_range_ang is None:
        z_min, z_max = float(z.min()), float(z.max()) + bin_width_ang
    else:
        z_min, z_max = z_range_ang
    if not z_min < z_max:
        raise ValueError(f"z range must have z_min < z_max; got ({z_min}, {z_max}).")

    if lateral_area_ang2 is None:
        lateral_area_ang2 = _cell_lateral_area_ang2(atoms)

    n_bins = max(1, int(np.ceil((z_max - z_min) / bin_width_ang)))
    edges = z_min + bin_width_ang * np.arange(n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])

    sl = scattering_lengths_fm or NEUTRON_SCATTERING_LENGTH_FM
    symbols = atoms.get_chemical_symbols()

    summed_b_fm = np.zeros(n_bins)
    for i, sym in enumerate(symbols):
        if sym not in sl:
            raise ValueError(
                f"No scattering length for species {sym!r}; pass scattering_lengths_fm."
            )
        idx = int(np.clip((z[i] - z_min) // bin_width_ang, 0, n_bins - 1))
        if z_min <= z[i] < z_max:
            summed_b_fm[idx] += sl[sym]

    bin_volume_ang3 = lateral_area_ang2 * bin_width_ang
    sld = (
        summed_b_fm
        * FEMTOMETERS_TO_ANGSTROMS
        / bin_volume_ang3
        * SLD_UNIT_CONVERSION_TO_E6_PER_A2
    )
    return SldProfile(
        z_centres_ang=centres,
        sld_e6_per_a2=sld,
        bin_width_ang=bin_width_ang,
    )


def _cell_lateral_area_ang2(atoms: Atoms) -> float:
    """In-plane area from the first two cell vectors (assumes z is the slab axis)."""
    a = np.asarray(atoms.cell[0])
    b = np.asarray(atoms.cell[1])
    return float(np.linalg.norm(np.cross(a, b)))
