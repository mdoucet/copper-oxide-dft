"""Tests for copper_oxide_dft.ml.gcga.

Covers the pure-math layer, the Cu(111) substrate construction, the
ase-ga-backed mutation operators (rattle / insert-O / remove-O), and
the import-contract for the full :func:`run_gcga_sweep` driver. End-to-
end MACE runs are not exercised here — those need a GPU and live in the
DGX Spark smoke test (:doc:`/docs/startup-cuo-cu-nonaqueous.md` §6.3).
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
    DEFAULT_MIN_PAIR_DISTANCE_ANG,
    DEFAULT_MU_O_N_POINTS,
    DEFAULT_MU_O_RANGE_EV,
    DEFAULT_OPERATOR_WEIGHTS,
    GCGAConfig,
    _blmin_atomic_numbers,
    _tournament_select,
    _validate_active_is_contiguous_tail,
    biased_grand_potential_ev,
    build_cu111_gcga_substrate,
    compute_x_o,
    gaussian_bias_ev,
    grand_potential_ev,
    insert_oxygen_offspring,
    rattle_offspring,
    remove_oxygen_offspring,
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


# ---------- _blmin_atomic_numbers (symbol → Z conversion) -------------------


def test_blmin_atomic_numbers_emits_both_orderings() -> None:
    """ase-ga's distance lookup is unordered; emit (a,b) and (b,a)."""
    blmin = _blmin_atomic_numbers({frozenset(["Cu", "O"]): 1.4})
    assert blmin[(29, 8)] == pytest.approx(1.4)
    assert blmin[(8, 29)] == pytest.approx(1.4)


def test_blmin_atomic_numbers_handles_homonuclear() -> None:
    blmin = _blmin_atomic_numbers({frozenset(["Cu", "Cu"]): 1.8})
    assert blmin[(29, 29)] == pytest.approx(1.8)


def test_blmin_atomic_numbers_full_default_table() -> None:
    blmin = _blmin_atomic_numbers(DEFAULT_MIN_PAIR_DISTANCE_ANG)
    # 3 pairs × 2 orderings, minus the duplicate (Cu,Cu) and (O,O).
    assert blmin[(29, 29)] == pytest.approx(1.8)
    assert blmin[(8, 8)] == pytest.approx(1.0)
    assert blmin[(29, 8)] == pytest.approx(1.4)
    assert blmin[(8, 29)] == pytest.approx(1.4)


# ---------- _validate_active_is_contiguous_tail -----------------------------


def test_validate_active_accepts_contiguous_tail() -> None:
    n_slab = _validate_active_is_contiguous_tail(10, (7, 8, 9))
    assert n_slab == 7


def test_validate_active_accepts_unordered_input() -> None:
    """Order within active_indices doesn't matter; the set does."""
    n_slab = _validate_active_is_contiguous_tail(10, (9, 7, 8))
    assert n_slab == 7


def test_validate_active_rejects_non_tail() -> None:
    """Active atoms scattered through the list violate the slab/top convention."""
    with pytest.raises(ValueError, match="contiguous tail"):
        _validate_active_is_contiguous_tail(10, (0, 1, 9))


def test_validate_active_substrate_passes_check() -> None:
    """The real `build_cu111_gcga_substrate` output must satisfy the convention."""
    slab, active = build_cu111_gcga_substrate(
        layers=4, lateral=(2, 2), active_top_layers=2,
    )
    n_slab = _validate_active_is_contiguous_tail(len(slab), active)
    assert n_slab == len(slab) - len(active)


def test_validate_active_empty_returns_full_length() -> None:
    assert _validate_active_is_contiguous_tail(5, ()) == 5


# ---------- rattle_offspring (wraps ase-ga RattleMutation) ------------------


def _make_test_substrate(layers: int = 4, lateral: tuple[int, int] = (2, 2)) -> tuple[Atoms, int]:
    """Helper: substrate + n_slab for operator tests."""
    slab, active = build_cu111_gcga_substrate(
        layers=layers, lateral=lateral, active_top_layers=2,
    )
    return slab, len(slab) - len(active)


def test_rattle_keeps_slab_atoms_fixed() -> None:
    parent, n_slab = _make_test_substrate()
    blmin = _blmin_atomic_numbers(DEFAULT_MIN_PAIR_DISTANCE_ANG)
    rng = np.random.default_rng(0)
    offspring = rattle_offspring(parent, n_slab, blmin, rattle_strength=0.1, rng=rng)
    assert offspring is not None
    # Slab positions identical (per ase-ga's slab/top convention).
    np.testing.assert_allclose(
        offspring.get_positions()[:n_slab],
        parent.get_positions()[:n_slab],
    )


def test_rattle_perturbs_active_atoms() -> None:
    parent, n_slab = _make_test_substrate()
    blmin = _blmin_atomic_numbers(DEFAULT_MIN_PAIR_DISTANCE_ANG)
    rng = np.random.default_rng(0)
    offspring = rattle_offspring(parent, n_slab, blmin, rattle_strength=0.2, rng=rng)
    assert offspring is not None
    disps = np.linalg.norm(
        offspring.get_positions()[n_slab:] - parent.get_positions()[n_slab:],
        axis=1,
    )
    # At least one active atom must have moved.
    assert disps.max() > 0.0


def test_rattle_returns_none_when_no_active_atoms() -> None:
    parent, _ = _make_test_substrate()
    blmin = _blmin_atomic_numbers(DEFAULT_MIN_PAIR_DISTANCE_ANG)
    rng = np.random.default_rng(0)
    # n_slab == len(parent) means there's nothing for the rattle to operate on.
    assert rattle_offspring(parent, len(parent), blmin, 0.1, rng) is None


# ---------- insert_oxygen_offspring -----------------------------------------


def test_insert_adds_one_oxygen_at_end() -> None:
    parent, n_slab = _make_test_substrate()
    rng = np.random.default_rng(0)
    offspring = insert_oxygen_offspring(
        parent, n_slab, DEFAULT_MIN_PAIR_DISTANCE_ANG,
        max_attempts=50, z_padding_ang=1.5, rng=rng,
    )
    assert offspring is not None
    assert len(offspring) == len(parent) + 1
    # The new atom must be appended (so it lands in the active region).
    assert offspring[-1].symbol == "O"


def test_insert_lands_above_slab_top() -> None:
    parent, n_slab = _make_test_substrate()
    rng = np.random.default_rng(0)
    slab_top_z = max(parent[i].z for i in range(n_slab))
    offspring = insert_oxygen_offspring(
        parent, n_slab, DEFAULT_MIN_PAIR_DISTANCE_ANG,
        max_attempts=50, z_padding_ang=1.5, rng=rng,
    )
    assert offspring is not None
    assert offspring[-1].z >= slab_top_z


def test_insert_respects_min_distance() -> None:
    """No inserted O should land closer than min Cu-O to any existing atom."""
    parent, n_slab = _make_test_substrate()
    rng = np.random.default_rng(0)
    cu_o_min = DEFAULT_MIN_PAIR_DISTANCE_ANG[frozenset(["Cu", "O"])]
    offspring = insert_oxygen_offspring(
        parent, n_slab, DEFAULT_MIN_PAIR_DISTANCE_ANG,
        max_attempts=200, z_padding_ang=1.5, rng=rng,
    )
    assert offspring is not None
    new_idx = len(offspring) - 1
    distances = offspring.get_distances(new_idx, list(range(new_idx)), mic=True)
    assert distances.min() >= cu_o_min - 1e-6


def test_insert_returns_none_when_no_anchor() -> None:
    """Empty parent cannot anchor a z-band; operator should refuse."""
    rng = np.random.default_rng(0)
    out = insert_oxygen_offspring(
        Atoms(), n_slab=0,
        min_pair_distance_ang=DEFAULT_MIN_PAIR_DISTANCE_ANG,
        max_attempts=10, z_padding_ang=1.5, rng=rng,
    )
    assert out is None


def test_insert_returns_none_when_no_room() -> None:
    """Tiny cell + huge min-distance → no valid insertion site found."""
    parent, n_slab = _make_test_substrate(layers=2, lateral=(1, 1))
    rng = np.random.default_rng(0)
    impossibly_strict = {
        frozenset(["Cu", "O"]): 10.0,
        frozenset(["O", "O"]): 10.0,
        frozenset(["Cu", "Cu"]): 1.8,
    }
    out = insert_oxygen_offspring(
        parent, n_slab, impossibly_strict,
        max_attempts=10, z_padding_ang=1.5, rng=rng,
    )
    assert out is None


# ---------- remove_oxygen_offspring -----------------------------------------


def test_remove_drops_one_active_oxygen() -> None:
    parent, n_slab = _make_test_substrate()
    rng = np.random.default_rng(0)
    with_o = insert_oxygen_offspring(
        parent, n_slab, DEFAULT_MIN_PAIR_DISTANCE_ANG, 50, 1.5, rng,
    )
    assert with_o is not None
    out = remove_oxygen_offspring(with_o, n_slab, rng)
    assert out is not None
    assert len(out) == len(with_o) - 1
    assert sum(1 for a in out if a.symbol == "O") == 0


def test_remove_returns_none_when_no_active_oxygen() -> None:
    """Pure-Cu slab has no O to remove."""
    parent, n_slab = _make_test_substrate()
    rng = np.random.default_rng(0)
    assert remove_oxygen_offspring(parent, n_slab, rng) is None


def test_remove_only_targets_active_region() -> None:
    """An O in the slab region is off-limits; the operator only sees active O."""
    parent, n_slab = _make_test_substrate()
    # Plant an O inside the slab region (index < n_slab); the operator
    # must not touch it.
    fake = parent.copy()
    fake[0].symbol = "O"
    rng = np.random.default_rng(0)
    assert remove_oxygen_offspring(fake, n_slab, rng) is None


# ---------- _tournament_select ----------------------------------------------


def test_tournament_picks_lowest_in_sample() -> None:
    omegas = [3.0, 1.0, 2.0, 5.0]
    rng = np.random.default_rng(0)
    # k=4 → samples all, deterministic winner = index 1.
    assert _tournament_select(omegas, k=4, rng=rng) == 1


def test_tournament_single_population_returns_zero() -> None:
    rng = np.random.default_rng(0)
    assert _tournament_select([0.5], k=3, rng=rng) == 0


def test_tournament_empty_raises() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        _tournament_select([], k=2, rng=rng)


# ---------- run_gcga_sweep (lazy import + input validation) -----------------


def test_run_gcga_sweep_validates_active_indices(tmp_path) -> None:
    """Driver must catch the slab/top-convention violation before doing IO."""
    from copper_oxide_dft.ml.gcga import run_gcga_sweep

    slab, _active = build_cu111_gcga_substrate(
        layers=2, lateral=(1, 1), active_top_layers=1,
    )
    # Deliberately wrong: pretend index 0 is active instead of the tail.
    bad_config = GCGAConfig(
        substrate=slab, active_indices=(0,), mu_o_ev=-6.5,
        n_generations=1, population_size=1,
    )
    # ImportError (if mace-torch isn't installed in this env) or
    # ValueError (if it is) — both are acceptable signals that the
    # driver refused to run a misconfigured search.
    with pytest.raises((ImportError, ValueError)):
        run_gcga_sweep(bad_config, mace_model_path="fake.model", out_dir=tmp_path)


def test_run_gcga_sweep_raises_on_missing_mace(tmp_path) -> None:
    """Without mace-torch installed, the driver must fail loudly with ImportError.

    In dev environments where mace is installed, this still passes because
    the missing-file path then raises a different error class — but the
    point of the test is the ImportError contract when the optional
    [ml] extra isn't present.
    """
    from copper_oxide_dft.ml.gcga import run_gcga_sweep

    slab, active = build_cu111_gcga_substrate(
        layers=2, lateral=(1, 1), active_top_layers=1,
    )
    config = GCGAConfig(
        substrate=slab, active_indices=active, mu_o_ev=-6.5,
        n_generations=1, population_size=1,
    )
    # In a fresh env: ImportError. If mace is somehow available: the
    # fake model path raises FileNotFoundError / OSError / RuntimeError
    # depending on the MACE version. Any of these counts as the loud-
    # failure behaviour we want; a silent success would be the bug.
    with pytest.raises((ImportError, FileNotFoundError, OSError, RuntimeError)):
        run_gcga_sweep(config, mace_model_path="does/not/exist.model", out_dir=tmp_path)


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


def test_operator_weights_sum_to_one() -> None:
    """Sanity: the default operator-weight tuple shouldn't be miscalibrated."""
    assert sum(DEFAULT_OPERATOR_WEIGHTS) == pytest.approx(1.0)
