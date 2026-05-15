"""Tests for copper_oxide_dft.che (Computational Hydrogen Electrode)."""

from __future__ import annotations

import math

import pytest

from copper_oxide_dft.che import (
    DEFAULT_TEMPERATURE_K,
    K_BOLTZMANN_EV_PER_K,
    AdsorbateState,
    PhaseEnergetics,
    ReferenceEnergetics,
    kT_ln10_ev,
    oxygen_chemical_potential_ev,
    phase_free_energy_per_cu_ev,
    proton_electron_chemical_potential_ev,
)

# ---- Boltzmann factor -------------------------------------------------------


def test_kT_ln10_matches_known_nernstian_slope_at_298K() -> None:
    """The famous 59 mV/pH-unit slope at standard temperature."""
    value = kT_ln10_ev(298.15)
    # Reference value: 8.617333262e-5 * 298.15 * ln(10) ≈ 0.05916 eV.
    assert value == pytest.approx(0.05916, abs=1e-4)


def test_kT_ln10_scales_linearly_with_temperature() -> None:
    a = kT_ln10_ev(100.0)
    b = kT_ln10_ev(300.0)
    assert b / a == pytest.approx(3.0, rel=1e-10)


# ---- proton-electron chemical potential -------------------------------------


def _trivial_references() -> ReferenceEnergetics:
    """Zero-energy refs (no ZPE/TS) for clean algebra in unit tests."""
    return ReferenceEnergetics(
        e_h2_ev=0.0,
        e_h2o_ev=0.0,
        zpe_h2_ev=0.0,
        zpe_h2o_ev=0.0,
        ts_h2_ev=0.0,
        ts_h2o_ev=0.0,
    )


def test_proton_electron_potential_is_zero_at_she_and_ph0_with_zero_refs() -> None:
    """At U=0, pH=0 with μ(H2)=0 the (H+ + e-) potential collapses to 0."""
    mu = proton_electron_chemical_potential_ev(
        _trivial_references(), u_she_v=0.0, ph=0.0
    )
    assert mu == pytest.approx(0.0, abs=1e-12)


def test_proton_electron_potential_decreases_with_u() -> None:
    """Each volt of U lowers μ(H+ + e-) by 1 eV."""
    refs = _trivial_references()
    mu0 = proton_electron_chemical_potential_ev(refs, u_she_v=0.0, ph=0.0)
    mu1 = proton_electron_chemical_potential_ev(refs, u_she_v=1.0, ph=0.0)
    assert mu1 - mu0 == pytest.approx(-1.0, abs=1e-12)


def test_proton_electron_potential_decreases_with_ph_by_nernstian_slope() -> None:
    """Each pH unit lowers μ(H+ + e-) by k_B T·ln(10)."""
    refs = _trivial_references()
    mu0 = proton_electron_chemical_potential_ev(refs, u_she_v=0.0, ph=0.0)
    mu7 = proton_electron_chemical_potential_ev(refs, u_she_v=0.0, ph=7.0)
    expected = -7.0 * K_BOLTZMANN_EV_PER_K * DEFAULT_TEMPERATURE_K * math.log(10.0)
    assert mu7 - mu0 == pytest.approx(expected, abs=1e-10)


# ---- oxygen chemical potential ----------------------------------------------


def test_oxygen_potential_at_she_and_ph0_equals_mu_h2o_minus_mu_h2() -> None:
    refs = ReferenceEnergetics(
        e_h2_ev=-1.0, e_h2o_ev=-5.0, zpe_h2_ev=0.0, zpe_h2o_ev=0.0,
        ts_h2_ev=0.0, ts_h2o_ev=0.0,
    )
    mu_o = oxygen_chemical_potential_ev(refs, u_she_v=0.0, ph=0.0)
    # μ(O) = μ(H2O) - μ(H2) = (-5) - (-1) = -4
    assert mu_o == pytest.approx(-4.0, abs=1e-12)


def test_oxygen_potential_rises_2eV_per_volt() -> None:
    """μ(O) slope w.r.t. U is +2 eV/V (two electrons per O atom)."""
    refs = _trivial_references()
    mu0 = oxygen_chemical_potential_ev(refs, u_she_v=0.0, ph=0.0)
    mu1 = oxygen_chemical_potential_ev(refs, u_she_v=1.0, ph=0.0)
    assert mu1 - mu0 == pytest.approx(2.0, abs=1e-10)


# ---- per-Cu free energy of a phase ------------------------------------------


def _literature_inputs() -> tuple[
    PhaseEnergetics, list[PhaseEnergetics], ReferenceEnergetics
]:
    """Experimental ΔG_f at 298 K cast into PhaseEnergetics for sanity tests."""
    references = ReferenceEnergetics(
        e_h2_ev=0.0, e_h2o_ev=-2.458,
        zpe_h2_ev=0.0, zpe_h2o_ev=0.0, ts_h2_ev=0.0, ts_h2o_ev=0.0,
    )
    cu = PhaseEnergetics(name="Cu", n_cu=1, n_o=0, e_dft_ev=0.0)
    cu2o = PhaseEnergetics(name="Cu2O", n_cu=2, n_o=1, e_dft_ev=-1.534)
    cuo = PhaseEnergetics(name="CuO", n_cu=1, n_o=1, e_dft_ev=-1.344)
    return cu, [cu, cu2o, cuo], references


def test_cu_metal_per_cu_free_energy_is_zero_everywhere() -> None:
    cu, _, refs = _literature_inputs()
    for u, ph in [(-1.0, 0.0), (0.0, 7.0), (1.0, 14.0)]:
        g = phase_free_energy_per_cu_ev(cu, cu, refs, u_she_v=u, ph=ph)
        assert g == pytest.approx(0.0, abs=1e-12)


def test_cu2o_more_stable_than_cuo_at_reducing_conditions() -> None:
    """Cu2O sits below CuO when μ(O) is low — i.e. at low U and low pH.

    CuO has a steeper -2 eV/V slope vs. U whereas Cu2O has -1 eV/V; the
    crossover happens at the experimental Cu2O/CuO Pourbaix boundary
    (roughly U ≈ +0.2 V at pH 7). Below the boundary, Cu2O wins.
    """
    cu, phases, refs = _literature_inputs()
    _, cu2o, cuo = phases
    g_cu2o = phase_free_energy_per_cu_ev(cu2o, cu, refs, u_she_v=-0.5, ph=0.0)
    g_cuo = phase_free_energy_per_cu_ev(cuo, cu, refs, u_she_v=-0.5, ph=0.0)
    assert g_cu2o < g_cuo


def test_cuo_more_stable_than_cu2o_at_oxidizing_conditions() -> None:
    """The opposite extreme: at high U + high pH, CuO is the lower-G phase."""
    cu, phases, refs = _literature_inputs()
    _, cu2o, cuo = phases
    g_cu2o = phase_free_energy_per_cu_ev(cu2o, cu, refs, u_she_v=1.0, ph=14.0)
    g_cuo = phase_free_energy_per_cu_ev(cuo, cu, refs, u_she_v=1.0, ph=14.0)
    assert g_cuo < g_cu2o


def test_oxide_per_cu_free_energy_decreases_with_potential() -> None:
    """Oxidizing potentials stabilize the oxide (negative slope of ΔG vs. U)."""
    cu, phases, refs = _literature_inputs()
    _, cu2o, _ = phases
    g_low = phase_free_energy_per_cu_ev(cu2o, cu, refs, u_she_v=-0.5, ph=0.0)
    g_high = phase_free_energy_per_cu_ev(cu2o, cu, refs, u_she_v=+0.5, ph=0.0)
    assert g_high < g_low
    # Slope is -2·n_o/n_cu = -1 V/V for Cu2O (n_o/n_cu = 0.5).
    assert g_high - g_low == pytest.approx(-1.0, abs=1e-10)


def test_at_minus_0p4_v_ph7_cu_metal_is_stable() -> None:
    """The user's experimental scenario: Cu metal is the stable solid phase."""
    cu, phases, refs = _literature_inputs()
    energies = {
        p.name: phase_free_energy_per_cu_ev(p, cu, refs, u_she_v=-0.4, ph=7.0)
        for p in phases
    }
    stable = min(energies, key=lambda k: energies[k])
    assert stable == "Cu"
    # Cu2O and CuO should be unstable by hundreds of meV.
    assert energies["Cu2O"] > 0.1
    assert energies["CuO"] > 0.1


def test_phase_free_energy_per_cu_rejects_zero_cu_phase() -> None:
    cu, _, refs = _literature_inputs()
    bad = PhaseEnergetics(name="O-island", n_cu=0, n_o=1, e_dft_ev=0.0)
    with pytest.raises(ValueError, match="n_cu"):
        phase_free_energy_per_cu_ev(bad, cu, refs, u_she_v=0.0, ph=0.0)


def test_phase_free_energy_per_cu_rejects_non_metal_reference() -> None:
    cu, phases, refs = _literature_inputs()
    _, cu2o, _ = phases
    with pytest.raises(ValueError, match="n_o=0"):
        # Using Cu2O as the per-Cu zero would be wrong.
        phase_free_energy_per_cu_ev(cu, cu2o, refs, u_she_v=0.0, ph=0.0)


# ---- ReferenceEnergetics dataclass behavior --------------------------------


# ---- Phase 4 v2: adsorbate states (clean Cu vs. covered surfaces) ----------


def _surface_inputs() -> tuple[AdsorbateState, ReferenceEnergetics]:
    refs = ReferenceEnergetics(
        e_h2_ev=0.0, e_h2o_ev=-2.458,
        zpe_h2_ev=0.0, zpe_h2o_ev=0.0, ts_h2_ev=0.0, ts_h2o_ev=0.0,
    )
    clean = AdsorbateState(
        name="Cu(111)", n_adsorbed_o=0, n_adsorbed_oh=0, e_dft_ev=-100.0
    )
    return clean, refs


def test_adsorbate_state_clean_relative_to_itself_is_zero() -> None:
    from copper_oxide_dft.che import adsorbate_state_relative_free_energy_ev

    clean, refs = _surface_inputs()
    for u, ph in [(-1.0, 0.0), (0.0, 7.0), (1.0, 14.0)]:
        assert adsorbate_state_relative_free_energy_ev(
            clean, clean, refs, u_she_v=u, ph=ph
        ) == pytest.approx(0.0, abs=1e-12)


def test_adsorbate_state_o_coverage_slope_minus_2_eV_per_V() -> None:
    """Each adsorbed O releases 2 electrons → ΔG drops 2 eV per +1 V."""
    from copper_oxide_dft.che import adsorbate_state_relative_free_energy_ev

    clean, refs = _surface_inputs()
    # One adsorbed O at some arbitrary slab energy. Slope per O is -2 eV/V.
    o_state = AdsorbateState(
        name="O on Cu(111)", n_adsorbed_o=1, n_adsorbed_oh=0,
        e_dft_ev=-103.0,
    )
    g0 = adsorbate_state_relative_free_energy_ev(
        o_state, clean, refs, u_she_v=0.0, ph=0.0
    )
    g1 = adsorbate_state_relative_free_energy_ev(
        o_state, clean, refs, u_she_v=1.0, ph=0.0
    )
    assert g1 - g0 == pytest.approx(-2.0, abs=1e-10)


def test_adsorbate_state_oh_coverage_slope_minus_1_eV_per_V() -> None:
    """Each adsorbed OH releases 1 electron → ΔG drops 1 eV per +1 V."""
    from copper_oxide_dft.che import adsorbate_state_relative_free_energy_ev

    clean, refs = _surface_inputs()
    oh_state = AdsorbateState(
        name="OH on Cu(111)", n_adsorbed_o=0, n_adsorbed_oh=1,
        e_dft_ev=-102.5,
    )
    g0 = adsorbate_state_relative_free_energy_ev(
        oh_state, clean, refs, u_she_v=0.0, ph=0.0
    )
    g1 = adsorbate_state_relative_free_energy_ev(
        oh_state, clean, refs, u_she_v=1.0, ph=0.0
    )
    assert g1 - g0 == pytest.approx(-1.0, abs=1e-10)


def test_adsorbate_state_oxidizing_potential_stabilizes_o_coverage() -> None:
    """At high U, the O-covered surface has lower ΔG than the clean one."""
    from copper_oxide_dft.che import adsorbate_state_relative_free_energy_ev

    clean, refs = _surface_inputs()
    o_state = AdsorbateState(
        name="O on Cu(111)", n_adsorbed_o=1, n_adsorbed_oh=0,
        e_dft_ev=-103.0,
    )
    # Pick a U high enough that even a weakly bound O wins.
    g = adsorbate_state_relative_free_energy_ev(
        o_state, clean, refs, u_she_v=+2.0, ph=0.0
    )
    assert g < 0.0


def test_adsorbate_state_n_proton_electron_pairs() -> None:
    """Each O carries 2 PCET; each OH carries 1."""
    state = AdsorbateState(
        name="mixed", n_adsorbed_o=2, n_adsorbed_oh=3, e_dft_ev=0.0
    )
    assert state.n_proton_electron_pairs == 2 * 2 + 1 * 3


def test_reference_energetics_default_zpe_ts_are_literature_values() -> None:
    """A user who passes only DFT total energies gets sensible defaults."""
    refs = ReferenceEnergetics(e_h2_ev=-1.0, e_h2o_ev=-2.0)
    assert refs.zpe_h2_ev > 0.0
    assert refs.ts_h2_ev > 0.0
    # mu_h2 = E + ZPE - TS
    assert refs.mu_h2_ev == pytest.approx(refs.e_h2_ev + refs.zpe_h2_ev - refs.ts_h2_ev)
    assert refs.mu_h2o_ev == pytest.approx(
        refs.e_h2o_ev + refs.zpe_h2o_ev - refs.ts_h2o_ev
    )
