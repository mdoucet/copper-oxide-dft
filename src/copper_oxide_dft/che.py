"""Computational Hydrogen Electrode (CHE) post-processing.

Nørskov's CHE replaces the chemical potential of a proton-electron pair with
that of (1/2) H2 at the standard hydrogen electrode (SHE) reference. For an
applied potential ``U`` (V vs. SHE) and pH:

    μ(H+ + e-) = (1/2)·μ(H2)  -  e·U  -  k_B T · ln(10) · pH

DFT total energies are computed at zero charge; the CHE shifts the
*post-processed* free energies by terms in U and pH. This is cheap, robust,
and answers thermodynamic questions ("which phase is stable at (U, pH)?")
without ever charging a slab. It does NOT answer kinetic questions, and does
NOT predict the structure of a charged interface — that needs ESM-FCP
(Phase 7).

For Cu / Cu2O / CuO the convenient reservoir is bulk Cu(metal) + H2O(l).
The chemical potential of an O atom in equilibrium with water under
electrochemical conditions follows from H2O <-> O + 2(H+ + e-):

    μ(O)(U, pH) = μ(H2O)  -  μ(H2)  +  2·e·U  +  2·k_B T · ln(10) · pH

The free energy of an oxide phase Cu_x O_y, referenced per Cu atom to bulk
Cu(metal), is then:

    ΔG_per_Cu(U, pH) = (G_phase - y·μ(O)(U, pH))/x  -  G_Cu_metal_per_atom

This is the function the Pourbaix diagram minimizes over the (U, pH) grid.

References:
    Nørskov et al., J. Phys. Chem. B 108, 17886 (2004) — canonical CHE paper.
    Hansen, Rossmeisl, Nørskov, PCCP 10, 3722 (2008) — Pourbaix construction.
    Persson et al., Phys. Rev. B 85, 235438 (2012) — solid-phase Pourbaix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Boltzmann constant in eV/K (CODATA 2018).
K_BOLTZMANN_EV_PER_K = 8.617333262e-5

# Standard temperature (K) for thermodynamic corrections.
DEFAULT_TEMPERATURE_K = 298.15

# Literature gas-phase thermodynamic corrections at 298.15 K used by the
# CHE community. ZPE values are PBE harmonic; -TS values are tabulated
# standard-state entropies from NIST Janaf (H2, O2) or from the
# Nørskov-school convention for H2O (gas at 0.035 bar to approximate
# liquid). Keeping these named and visible: if the user wants to use a
# different convention they can pass overrides.

ZPE_H2_EV = 0.27
"""H2 zero-point energy (eV). Nørskov 2004."""

ZPE_H2O_EV = 0.56
"""H2O zero-point energy (eV). Standard CHE value."""

ZPE_O2_EV = 0.10
"""O2 zero-point energy (eV). Not used by the H2O-referenced CHE, kept for
reference and for the OER/ORR overpotential analysis in later phases."""

TS_H2_EV = 0.40
"""TΔS for H2(g) at 298.15 K, 1 bar (eV). NIST Janaf."""

TS_H2O_EV = 0.67
"""TΔS for H2O at 298.15 K (eV). Computed at 0.035 bar (= vapor pressure
of liquid water) so the gas-phase H2O reference approximates liquid."""

TS_O2_EV = 0.64
"""TΔS for O2(g) at 298.15 K, 1 bar (eV). NIST Janaf."""


@dataclass(frozen=True)
class PhaseEnergetics:
    """DFT-computed energetics for a Cu-containing phase, per formula unit.

    Use one ``PhaseEnergetics`` per solid phase in the Pourbaix construction.
    Cu (metal) has ``n_o=0`` and serves as the per-Cu reference. Cu2O is
    ``n_cu=2, n_o=1``; CuO is ``n_cu=1, n_o=1``.

    For bulk solids the ZPE and TS contributions are typically <0.05 eV per
    formula unit and conventionally dropped; defaults are zero. For adsorbed
    species on slabs (Phase 4 onward) you'll want to include them — the
    Hansen 2008 supplementary lists tabulated values.

    Attributes:
        name: Human-readable label used by the Pourbaix plotter.
        n_cu: Cu atoms per formula unit.
        n_o: O atoms per formula unit.
        e_dft_ev: Total DFT energy per formula unit, eV.
        zpe_ev: Zero-point energy correction, eV.
        ts_ev: T·S thermal entropy correction, eV.
    """

    name: str
    n_cu: int
    n_o: int
    e_dft_ev: float
    zpe_ev: float = 0.0
    ts_ev: float = 0.0

    @property
    def free_energy_ev(self) -> float:
        """Free energy per formula unit: G = E_DFT + ZPE - TS."""
        return self.e_dft_ev + self.zpe_ev - self.ts_ev


@dataclass(frozen=True)
class ReferenceEnergetics:
    """Reference chemical potentials for H2(g) and H2O(l) at one level of theory.

    The CHE needs μ(H2) and μ(H2O) computed at the *same* DFT level as the
    phase energies. Don't mix-and-match cutoffs, pseudopotentials, or U
    values across the reference and the phases — total energies do not
    cancel cleanly otherwise.

    Compute these from single-molecule QE runs (see
    :func:`copper_oxide_dft.structure_builder.build_reference_h2` and
    ``build_reference_h2o``) and pass ``e_dft_ev`` from the parsed output.

    Attributes:
        e_h2_ev: Total DFT energy of an isolated H2 molecule, eV.
        e_h2o_ev: Total DFT energy of an isolated H2O molecule, eV.
        zpe_h2_ev: ZPE of H2 (default :data:`ZPE_H2_EV`).
        zpe_h2o_ev: ZPE of H2O (default :data:`ZPE_H2O_EV`).
        ts_h2_ev: TS of H2 at T (default :data:`TS_H2_EV` at 298 K).
        ts_h2o_ev: TS of H2O at T (default :data:`TS_H2O_EV` at 298 K).
    """

    e_h2_ev: float
    e_h2o_ev: float
    zpe_h2_ev: float = ZPE_H2_EV
    zpe_h2o_ev: float = ZPE_H2O_EV
    ts_h2_ev: float = TS_H2_EV
    ts_h2o_ev: float = TS_H2O_EV

    @property
    def mu_h2_ev(self) -> float:
        """Free-energy chemical potential of H2(g): E + ZPE - TS."""
        return self.e_h2_ev + self.zpe_h2_ev - self.ts_h2_ev

    @property
    def mu_h2o_ev(self) -> float:
        """Free-energy chemical potential of H2O(l-approx): E + ZPE - TS."""
        return self.e_h2o_ev + self.zpe_h2o_ev - self.ts_h2o_ev


def kT_ln10_ev(temperature_k: float = DEFAULT_TEMPERATURE_K) -> float:
    """k_B T · ln(10) at given temperature, in eV.

    The factor that multiplies pH in CHE expressions. At 298.15 K this
    is ≈ 0.0592 eV, the famous 59 mV/pH-unit Nernstian slope.
    """
    return K_BOLTZMANN_EV_PER_K * temperature_k * math.log(10.0)


def proton_electron_chemical_potential_ev(
    references: ReferenceEnergetics,
    u_she_v: float,
    ph: float,
    temperature_k: float = DEFAULT_TEMPERATURE_K,
) -> float:
    """μ(H+ + e-) under CHE at (U vs. SHE, pH).

    Formula: ``μ(H+ + e-) = (1/2)·μ(H2) - e·U - k_B T·ln(10)·pH``.

    Note: ``u_she_v`` is in volts; the ``-e·U`` term in the formula becomes
    ``-1·U`` numerically when U is in volts and the result is in eV,
    because the elementary charge is exactly 1 in those units.

    Args:
        references: H2 and H2O DFT energetics.
        u_she_v: Applied potential vs. standard hydrogen electrode, V.
        ph: Solution pH.
        temperature_k: Temperature in Kelvin.

    Returns:
        Chemical potential of the proton-electron pair, eV.
    """
    return 0.5 * references.mu_h2_ev - u_she_v - kT_ln10_ev(temperature_k) * ph


def oxygen_chemical_potential_ev(
    references: ReferenceEnergetics,
    u_she_v: float,
    ph: float,
    temperature_k: float = DEFAULT_TEMPERATURE_K,
) -> float:
    """μ(O) referenced to H2O under CHE at (U, pH).

    From the equilibrium H2O <-> O + 2(H+ + e-):

        μ(O) = μ(H2O) - 2·μ(H+ + e-)
             = μ(H2O) - μ(H2) + 2·U + 2·k_B T·ln(10)·pH

    The slope vs. U (+2 eV/V) reflects two electron transfers per O atom
    deposited from water.

    Args:
        references: H2 and H2O DFT energetics.
        u_she_v: Applied potential vs. SHE, V.
        ph: Solution pH.
        temperature_k: Temperature, K.

    Returns:
        Chemical potential of an O atom in equilibrium with water, eV.
    """
    mu_h_pair = proton_electron_chemical_potential_ev(
        references, u_she_v, ph, temperature_k
    )
    return references.mu_h2o_ev - 2.0 * mu_h_pair


@dataclass(frozen=True)
class AdsorbateState:
    """A specific coverage state on a fixed Cu(111) slab.

    Used to build the Phase-4-v2 *surface* Pourbaix: each state is a
    (n_o, n_oh) combination on the same supercell, and we compare
    their free energies at (U, pH) to find the lowest-G surface
    termination.

    The reference is always the bare clean slab on the same supercell;
    do NOT mix states from different cell sizes in one diagram (the
    absolute energies don't subtract cleanly).

    Adsorbates are taken from the H2O reservoir:

    * O: ``Cu* + H2O -> Cu*-O + 2(H+ + e-)``      (n_PCET = 2 per O)
    * OH: ``Cu* + H2O -> Cu*-OH + (H+ + e-)``     (n_PCET = 1 per OH)

    Attributes:
        name: Human-readable label (e.g. "clean", "1/9 ML O").
        n_adsorbed_o: Atomic O adsorbates added on top of the clean slab.
        n_adsorbed_oh: OH groups adsorbed on top of the clean slab.
        e_dft_ev: Total DFT energy of THIS slab (eV). Includes all
            adsorbates and any per-atom corrections; the
            ZPE/TS contributions for adsorbed O/OH belong here as
            ``zpe_ev`` and ``ts_ev``.
        zpe_ev: ZPE correction for all adsorbates on the slab (eV).
            Literature values for adsorbed O: ~0.05; OH: ~0.36 (per
            adsorbate) — multiply by the count when filling in.
        ts_ev: T·S thermal entropy for adsorbates (eV). Typically
            small (<0.05 eV) for tightly bound adsorbates at 298 K.
    """

    name: str
    n_adsorbed_o: int
    n_adsorbed_oh: int
    e_dft_ev: float
    zpe_ev: float = 0.0
    ts_ev: float = 0.0

    @property
    def free_energy_ev(self) -> float:
        """G of the entire slab: E_DFT + sum(ZPE) - sum(TS)."""
        return self.e_dft_ev + self.zpe_ev - self.ts_ev

    @property
    def n_proton_electron_pairs(self) -> int:
        """Total (H+ + e-) pairs released from clean -> this state.

        Each O comes from a water and releases 2 PCET; each OH releases 1.
        """
        return 2 * self.n_adsorbed_o + 1 * self.n_adsorbed_oh


def adsorbate_state_relative_free_energy_ev(
    state: AdsorbateState,
    clean_slab: AdsorbateState,
    water_reference: ReferenceEnergetics,
    u_she_v: float,
    ph: float,
    temperature_k: float = DEFAULT_TEMPERATURE_K,
) -> float:
    """ΔG of an adsorbate state relative to the clean slab at (U, pH).

    Both adsorbates are referenced to the H2O reservoir; their chemical
    potentials already carry the (U, pH) dependence::

        μ(O)(U, pH)  = μ(H2O) − μ(H2) + 2·eU + 2·k_B T·ln10·pH
        μ(OH)(U, pH) = μ(H2O) − μ(H+ + e−)
                     = μ(H2O) − (1/2)μ(H2) + eU + k_B T·ln10·pH

    So the surface free-energy difference is::

        ΔG = G_state − G_clean − n_o·μ(O) − n_oh·μ(OH)

    There is no extra CHE shift on top — that's already inside μ(O)
    and μ(OH). The clean slab gives ΔG = 0 by construction (all
    coverage counts zero).

    Args:
        state: Coverage state to evaluate.
        clean_slab: Bare Cu(111) reference at the same cell size.
        water_reference: H2 / H2O DFT energetics.
        u_she_v: Applied potential vs. SHE (V).
        ph: Solution pH.
        temperature_k: Temperature (K).

    Returns:
        ΔG of ``state`` relative to ``clean_slab`` at (U, pH), in eV
        (absolute, not per-area; convert if comparing across cells).
    """
    # μ(O) and μ(OH) as functions of (U, pH).
    mu_o = oxygen_chemical_potential_ev(water_reference, u_she_v, ph, temperature_k)
    # μ(OH) = μ(H2O) - μ(H+ + e-), i.e. removing one proton-electron pair
    # from a water molecule.
    mu_h_pair = proton_electron_chemical_potential_ev(
        water_reference, u_she_v, ph, temperature_k
    )
    mu_oh = water_reference.mu_h2o_ev - mu_h_pair

    delta_g = (
        state.free_energy_ev
        - clean_slab.free_energy_ev
        - state.n_adsorbed_o * mu_o
        - state.n_adsorbed_oh * mu_oh
    )
    return delta_g


def phase_free_energy_per_cu_ev(
    phase: PhaseEnergetics,
    cu_metal_reference: PhaseEnergetics,
    water_reference: ReferenceEnergetics,
    u_she_v: float,
    ph: float,
    temperature_k: float = DEFAULT_TEMPERATURE_K,
) -> float:
    """Per-Cu free energy of a phase, referenced to bulk Cu + H2O.

    ``Cu(metal) + (n_o/n_cu) H2O <-> (1/n_cu)·Cu_xO_y + (2 n_o/n_cu)·(H+ + e-)``

    Per-Cu free energy: ::

        ΔG_per_Cu(U, pH) = (G_phase - n_o · μ(O)(U, pH)) / n_cu
                          - G_Cu_metal / n_cu_in_reference

    Pure Cu metal gives 0 by construction. Oxide phases give a value that
    decreases linearly with U (slope -2·n_o/n_cu eV/V) — oxidizing
    potentials stabilize oxides.

    Args:
        phase: Phase to evaluate (oxide or metal).
        cu_metal_reference: Bulk Cu metal energetics, used as per-Cu zero.
        water_reference: H2 and H2O DFT energetics for μ(O).
        u_she_v: Potential vs. SHE, V.
        ph: Solution pH.
        temperature_k: Temperature, K.

    Returns:
        Per-Cu free energy of ``phase`` relative to Cu metal, eV/Cu.

    Raises:
        ValueError: If ``phase.n_cu`` is zero, or if
            ``cu_metal_reference.n_cu`` is zero or
            ``cu_metal_reference.n_o`` is nonzero (not pure Cu metal).
    """
    if phase.n_cu <= 0:
        raise ValueError(
            f"Phase {phase.name!r} has n_cu={phase.n_cu}; "
            "per-Cu free energy is undefined."
        )
    if cu_metal_reference.n_cu <= 0 or cu_metal_reference.n_o != 0:
        raise ValueError(
            f"Cu-metal reference must have n_cu>0 and n_o=0; got "
            f"n_cu={cu_metal_reference.n_cu}, n_o={cu_metal_reference.n_o}."
        )

    mu_o = oxygen_chemical_potential_ev(water_reference, u_she_v, ph, temperature_k)
    g_oxide_per_cu = (phase.free_energy_ev - phase.n_o * mu_o) / phase.n_cu
    g_cu_per_cu = cu_metal_reference.free_energy_ev / cu_metal_reference.n_cu
    return g_oxide_per_cu - g_cu_per_cu
