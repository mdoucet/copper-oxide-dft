"""Tests for copper_oxide_dft.ml.gcga (pure-math + substrate parts).

The GOCIA driver (:func:`run_gcga_sweep`) is exercised only by the
ImportError / NotImplementedError contract — actually running it needs
the [ml] extras + DGX Spark.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms

from copper_oxide_dft.ml.gcga import (
    DEFAULT_ACTIVE_TOP_LAYERS,
    DEFAULT_BIASED_AMPLITUDE_EV,
    DEFAULT_BIASED_SIGMA,
    DEFAULT_BIASED_X_O_RANGE,
    DEFAULT_LATERAL_GCGA,
    DEFAULT_LAYERS_GCGA,
    DEFAULT_MU_O_N_POINTS,
    DEFAULT_MU_O_RANGE_EV,
    GCGAConfig,
    biased_grand_potential_ev,
    build_cu111_gcga_substrate,
    compute_x_o,
    gaussian_bias_ev,
    grand_potential_ev,
)
from copper_oxide_dft.structure_builder import (
    build_bulk_cu,
    build_bulk_cu2o,
    build_bulk_cuo,
)

# ---------- compute_x_o -------------------------------------------------------


def test_x_o_pure_cu_is_zero() -> None:
    assert compute_x_o(build_bulk_cu()) == 0.0


def test_x_o_cu2o_is_one_third() -> None:
    np.testing.assert_allclose(compute_x_o(build_bulk_cu2o()), 2.0 / 6.0)


def test_x_o_cuo_is_one_half() -> None:
    np.testing.assert_allclose(compute_x_o(build_bulk_cuo()), 4.0 / 8.0)


def test_x_o_empty_returns_zero() -> None:
    assert compute_x_o(Atoms()) == 0.0


def test_x_o_pure_oxygen_is_one() -> None:
    atoms = Atoms("O2", positions=[(0, 0, 0), (1, 0, 0)], cell=[10, 10, 10], pbc=True)
    assert compute_x_o(atoms) == 1.0


# ---------- grand_potential_ev ------------------------------------------------


def test_grand_potential_with_no_oxygen_is_just_energy() -> None:
    e = -10.5
    assert grand_potential_ev(e, build_bulk_cu(), mu_o_ev=-6.5) == pytest.approx(e)


def test_grand_potential_subtracts_mu_o_times_n_o() -> None:
    # 2 O atoms, μ_O = -6.0 eV: Ω = E - μ_O · N_O = E - (-6.0 · 2) = E + 12.0
    cu2o = build_bulk_cu2o()
    e = -100.0
    assert grand_potential_ev(e, cu2o, mu_o_ev=-6.0) == pytest.approx(-100.0 + 12.0)


def test_grand_potential_linear_in_mu_o() -> None:
    """Doubling |μ_O| doubles the correction (for nonzero N_O)."""
    cu2o = build_bulk_cu2o()
    e = -100.0
    omega1 = grand_potential_ev(e, cu2o, mu_o_ev=-3.0)
    omega2 = grand_potential_ev(e, cu2o, mu_o_ev=-6.0)
    np.testing.assert_allclose(omega2 - e, 2.0 * (omega1 - e), rtol=1e-9)


# ---------- gaussian_bias_ev --------------------------------------------------


def test_bias_no_centers_is_zero() -> None:
    assert gaussian_bias_ev(0.5, centers=[], amplitude_ev=1.0, sigma=0.1) == 0.0


def test_bias_at_center_equals_amplitude() -> None:
    bias = gaussian_bias_ev(0.5, centers=[0.5], amplitude_ev=0.7, sigma=0.05)
    np.testing.assert_allclose(bias, 0.7)


def test_bias_at_three_sigma_is_small() -> None:
    """Three sigma away: exp(-9/2) ≈ 0.011."""
    sigma = 0.05
    bias = gaussian_bias_ev(0.5 + 3 * sigma, centers=[0.5], amplitude_ev=1.0, sigma=sigma)
    assert bias < 0.02
    assert bias > 0.0


def test_bias_sums_over_centers() -> None:
    """Two coincident centers: bias = 2 × amplitude at the center."""
    bias = gaussian_bias_ev(0.5, centers=[0.5, 0.5], amplitude_ev=0.3, sigma=0.05)
    np.testing.assert_allclose(bias, 0.6)


def test_bias_is_symmetric_in_x_around_center() -> None:
    sigma, c = 0.05, 0.5
    left = gaussian_bias_ev(c - 0.1, [c], 1.0, sigma)
    right = gaussian_bias_ev(c + 0.1, [c], 1.0, sigma)
    np.testing.assert_allclose(left, right)


def test_bias_rejects_non_positive_sigma() -> None:
    with pytest.raises(ValueError):
        gaussian_bias_ev(0.5, [0.5], 1.0, sigma=0.0)
    with pytest.raises(ValueError):
        gaussian_bias_ev(0.5, [0.5], 1.0, sigma=-0.05)


# ---------- biased_grand_potential_ev ----------------------------------------


def test_biased_omega_equals_unbiased_when_no_centers() -> None:
    cu2o = build_bulk_cu2o()
    config = GCGAConfig(
        substrate=Atoms(), active_indices=(), mu_o_ev=-6.0,
        bias_centers=(),
    )
    omega_biased = biased_grand_potential_ev(-100.0, cu2o, config)
    omega_plain = grand_potential_ev(-100.0, cu2o, -6.0)
    assert omega_biased == omega_plain


def test_biased_omega_adds_gaussian_at_x_o() -> None:
    cu2o = build_bulk_cu2o()
    x = compute_x_o(cu2o)
    config = GCGAConfig(
        substrate=Atoms(), active_indices=(), mu_o_ev=-6.0,
        bias_centers=(x,), bias_amplitude_ev=0.5, bias_sigma=0.05,
    )
    omega_biased = biased_grand_potential_ev(-100.0, cu2o, config)
    omega_plain = grand_potential_ev(-100.0, cu2o, -6.0)
    np.testing.assert_allclose(omega_biased - omega_plain, 0.5)


# ---------- build_cu111_gcga_substrate ---------------------------------------


def test_substrate_atom_count_matches_layers_x_lateral() -> None:
    """12 layers × 4×4 lateral = 192 atoms."""
    slab, _ = build_cu111_gcga_substrate(
        layers=12, lateral=(4, 4), active_top_layers=6,
    )
    assert len(slab) == 12 * 4 * 4


def test_substrate_active_indices_are_top_n_layers() -> None:
    slab, active = build_cu111_gcga_substrate(
        layers=4, lateral=(2, 2), active_top_layers=2,
    )
    # 4 layers × 2×2 = 16 atoms; top 2 layers × 4 atoms = 8 active.
    assert len(slab) == 16
    assert len(active) == 8
    # Active atoms have z greater than all non-active atoms.
    active_set = set(active)
    active_zs = [slab[i].z for i in range(len(slab)) if i in active_set]
    inactive_zs = [slab[i].z for i in range(len(slab)) if i not in active_set]
    assert min(active_zs) > max(inactive_zs)


def test_substrate_carries_fix_constraint_on_bottom_layers() -> None:
    slab, _ = build_cu111_gcga_substrate(
        layers=4, lateral=(2, 2), active_top_layers=2,
    )
    constraints = slab.constraints
    assert any(isinstance(c, FixAtoms) for c in constraints)
    fixed = next(c.index for c in constraints if isinstance(c, FixAtoms))
    # Fixed atoms are the bottom 2 layers = 8 atoms.
    assert len(fixed) == 8


def test_substrate_respects_lattice_a_override() -> None:
    """Passing the PBE-relaxed `a` produces a larger cell than the default."""
    slab_exp, _ = build_cu111_gcga_substrate(
        layers=2, lateral=(2, 2), active_top_layers=1, lattice_a_ang=3.615,
    )
    slab_relaxed, _ = build_cu111_gcga_substrate(
        layers=2, lateral=(2, 2), active_top_layers=1, lattice_a_ang=3.6577,
    )
    # Lateral cell vectors scale linearly with a.
    np.testing.assert_allclose(
        np.linalg.norm(slab_relaxed.cell[0]) / np.linalg.norm(slab_exp.cell[0]),
        3.6577 / 3.615,
        rtol=1e-6,
    )


def test_substrate_rejects_bad_layer_args() -> None:
    with pytest.raises(ValueError):
        build_cu111_gcga_substrate(layers=0, lateral=(2, 2), active_top_layers=1)
    with pytest.raises(ValueError):
        build_cu111_gcga_substrate(layers=4, lateral=(2, 2), active_top_layers=0)
    with pytest.raises(ValueError):
        build_cu111_gcga_substrate(layers=4, lateral=(2, 2), active_top_layers=10)


# ---------- run_gcga_sweep (lazy import smoke test) --------------------------


def test_run_gcga_sweep_raises_before_gocia_pinned(tmp_path) -> None:
    """First-run-on-DGX behavior: raise loudly instead of silently doing nothing.

    Only meaningful if GOCIA is installed; otherwise we get ImportError.
    Either signal is acceptable (and both are tested implicitly here).
    """
    from copper_oxide_dft.ml.gcga import run_gcga_sweep

    slab, active = build_cu111_gcga_substrate(
        layers=2, lateral=(1, 1), active_top_layers=1,
    )
    config = GCGAConfig(substrate=slab, active_indices=active, mu_o_ev=-6.5,
                       n_generations=1, population_size=1)
    with pytest.raises((ImportError, NotImplementedError)):
        run_gcga_sweep(config, mace_model_path="fake.model", out_dir=tmp_path)


# ---------- constant sanity ---------------------------------------------------


def test_defaults_match_pivot_doc() -> None:
    assert DEFAULT_MU_O_RANGE_EV == (-7.0, -6.0)
    assert DEFAULT_MU_O_N_POINTS == 11
    assert DEFAULT_BIASED_X_O_RANGE == (0.32, 1.0)
    assert DEFAULT_LAYERS_GCGA == 12
    assert DEFAULT_ACTIVE_TOP_LAYERS == 6
    assert DEFAULT_LATERAL_GCGA == (4, 4)
    assert DEFAULT_BIASED_AMPLITUDE_EV == 0.5
    assert DEFAULT_BIASED_SIGMA == 0.05
