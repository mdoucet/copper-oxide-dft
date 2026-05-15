"""Tests for copper_oxide_dft.pourbaix."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # headless: needed for CI / Frontier login nodes

from copper_oxide_dft.che import AdsorbateState, PhaseEnergetics, ReferenceEnergetics
from copper_oxide_dft.pourbaix import (
    adsorbate_phase_diagram,
    phase_diagram,
    plot_diagram,
)


def _literature_setup() -> tuple[
    list[PhaseEnergetics], PhaseEnergetics, ReferenceEnergetics
]:
    """Literature ΔG_f inputs (matches the CLI defaults)."""
    cu = PhaseEnergetics(name="Cu(metal)", n_cu=1, n_o=0, e_dft_ev=0.0)
    cu2o = PhaseEnergetics(name="Cu2O", n_cu=2, n_o=1, e_dft_ev=-1.534)
    cuo = PhaseEnergetics(name="CuO", n_cu=1, n_o=1, e_dft_ev=-1.344)
    refs = ReferenceEnergetics(
        e_h2_ev=0.0, e_h2o_ev=-2.458,
        zpe_h2_ev=0.0, zpe_h2o_ev=0.0, ts_h2_ev=0.0, ts_h2o_ev=0.0,
    )
    return [cu, cu2o, cuo], cu, refs


def test_phase_diagram_grid_shapes_match_options() -> None:
    phases, cu, refs = _literature_setup()
    diagram = phase_diagram(
        phases, cu, refs,
        u_range_v=(-1.0, 1.0), u_steps=21,
        ph_range=(0.0, 14.0), ph_steps=15,
    )
    assert diagram.u_grid_v.shape == (21,)
    assert diagram.ph_grid.shape == (15,)
    assert diagram.free_energies_per_cu_ev.shape == (3, 15, 21)
    assert diagram.stable_phase_index.shape == (15, 21)


def test_phase_diagram_marks_cu_stable_at_minus_0p4_v_ph7() -> None:
    """The user's experimental scenario: 3 nm CuO at -0.4 V / pH 7 reduces to Cu."""
    phases, cu, refs = _literature_setup()
    diagram = phase_diagram(phases, cu, refs)
    assert diagram.stable_phase_at(u_she_v=-0.4, ph=7.0) == "Cu(metal)"


def test_phase_diagram_marks_cuo_stable_at_oxidizing_alkaline_corner() -> None:
    """High U, high pH should give CuO as the stable phase."""
    phases, cu, refs = _literature_setup()
    diagram = phase_diagram(phases, cu, refs)
    assert diagram.stable_phase_at(u_she_v=1.0, ph=14.0) == "CuO"


def test_phase_diagram_has_three_distinct_regions_in_default_window() -> None:
    """Cu / Cu2O / CuO should all appear somewhere in the default (U, pH) box."""
    phases, cu, refs = _literature_setup()
    diagram = phase_diagram(phases, cu, refs)
    unique = set(np.unique(diagram.stable_phase_index).tolist())
    # phase 0 = Cu, 1 = Cu2O, 2 = CuO
    assert unique == {0, 1, 2}


def test_phase_diagram_rejects_empty_phase_list() -> None:
    _, cu, refs = _literature_setup()
    with pytest.raises(ValueError, match="at least one phase"):
        phase_diagram([], cu, refs)


def test_phase_diagram_rejects_degenerate_grid() -> None:
    phases, cu, refs = _literature_setup()
    with pytest.raises(ValueError, match=">= 2"):
        phase_diagram(phases, cu, refs, u_steps=1)


def test_plot_diagram_returns_axes_with_correct_axis_labels() -> None:
    import matplotlib.pyplot as plt

    phases, cu, refs = _literature_setup()
    diagram = phase_diagram(
        phases, cu, refs, u_steps=11, ph_steps=11
    )
    fig, ax = plt.subplots()
    ax = plot_diagram(diagram, ax=ax, mark_point=(-0.4, 7.0))
    assert ax.get_xlabel() == "U vs. SHE (V)"
    assert ax.get_ylabel() == "pH"
    title = ax.get_title()
    assert "Cu" in title
    plt.close(fig)


def test_plot_diagram_creates_figure_when_no_axes_passed() -> None:
    import matplotlib.pyplot as plt

    phases, cu, refs = _literature_setup()
    diagram = phase_diagram(phases, cu, refs, u_steps=11, ph_steps=11)
    ax = plot_diagram(diagram)
    assert ax.figure is not None
    plt.close(ax.figure)


# ---- adsorbate phase diagram (Phase 4 v2) ----------------------------------


def _adsorbate_setup() -> tuple[
    list[AdsorbateState], AdsorbateState, ReferenceEnergetics
]:
    refs = ReferenceEnergetics(
        e_h2_ev=0.0, e_h2o_ev=-2.458,
        zpe_h2_ev=0.0, zpe_h2o_ev=0.0, ts_h2_ev=0.0, ts_h2o_ev=0.0,
    )
    clean = AdsorbateState(
        name="clean", n_adsorbed_o=0, n_adsorbed_oh=0, e_dft_ev=-100.0
    )
    # Modest binding energies so the diagram has both regions in (-1, +1) V:
    # too-strong binding makes O stable everywhere; too-weak makes clean win
    # everywhere. The values below produce a Cu/O/OH region structure that
    # mimics the experimental Cu(111) wet-electrochemistry picture.
    o_state = AdsorbateState(
        name="1 O", n_adsorbed_o=1, n_adsorbed_oh=0, e_dft_ev=-101.5
    )
    oh_state = AdsorbateState(
        name="1 OH", n_adsorbed_o=0, n_adsorbed_oh=1, e_dft_ev=-100.7
    )
    return [clean, o_state, oh_state], clean, refs


def test_adsorbate_phase_diagram_clean_stable_at_reducing_low_pH() -> None:
    states, clean, refs = _adsorbate_setup()
    diagram = adsorbate_phase_diagram(states, clean, refs)
    assert diagram.stable_phase_at(u_she_v=-0.8, ph=0.0) == "clean"


def test_adsorbate_phase_diagram_o_stable_at_oxidizing_high_pH() -> None:
    states, clean, refs = _adsorbate_setup()
    diagram = adsorbate_phase_diagram(states, clean, refs)
    assert diagram.stable_phase_at(u_she_v=+1.0, ph=14.0) == "1 O"


def test_adsorbate_phase_diagram_rejects_empty_states() -> None:
    _, clean, refs = _adsorbate_setup()
    with pytest.raises(ValueError, match="at least one adsorbate state"):
        adsorbate_phase_diagram([], clean, refs)


def test_adsorbate_phase_diagram_rejects_degenerate_grid() -> None:
    states, clean, refs = _adsorbate_setup()
    with pytest.raises(ValueError, match=">= 2"):
        adsorbate_phase_diagram(states, clean, refs, u_steps=1)
