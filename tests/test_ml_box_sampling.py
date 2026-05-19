"""Tests for copper_oxide_dft.ml.box_sampling."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from copper_oxide_dft.ml.box_sampling import (
    CU_O_CONNECTIVITY_CUTOFF_ANG,
    DEFAULT_INSERT_MIN_DISTANCE_ANG,
    BoxSamplingConfig,
    PerturbationResult,
    apply_hookean_repair,
    enforce_cu_o_connectivity,
    perturb_structure,
    sample_batch,
)
from copper_oxide_dft.structure_builder import (
    build_bulk_cu,
    build_bulk_cu2o,
    build_bulk_cuo,
)

# ---------- enforce_cu_o_connectivity ----------------------------------------


def test_connectivity_passes_when_no_oxygen() -> None:
    atoms = build_bulk_cu()
    assert enforce_cu_o_connectivity(atoms, CU_O_CONNECTIVITY_CUTOFF_ANG) is True


def test_connectivity_passes_for_intact_cu2o() -> None:
    assert (
        enforce_cu_o_connectivity(build_bulk_cu2o(), CU_O_CONNECTIVITY_CUTOFF_ANG)
        is True
    )


def test_connectivity_fails_when_oxygen_is_isolated() -> None:
    # One Cu near origin, one O 5 Å away — far beyond the 2.8 Å cutoff.
    atoms = Atoms(
        symbols=["Cu", "O"],
        positions=[(0.0, 0.0, 0.0), (5.0, 0.0, 0.0)],
        cell=[20.0, 20.0, 20.0],
        pbc=True,
    )
    assert enforce_cu_o_connectivity(atoms, CU_O_CONNECTIVITY_CUTOFF_ANG) is False


def test_connectivity_fails_when_only_oxygen() -> None:
    atoms = Atoms("O", positions=[(0.0, 0.0, 0.0)], cell=[10, 10, 10], pbc=True)
    assert enforce_cu_o_connectivity(atoms, CU_O_CONNECTIVITY_CUTOFF_ANG) is False


def test_connectivity_uses_minimum_image() -> None:
    # Cu and O on opposite corners of a small cell — distance via MIC is tiny.
    atoms = Atoms(
        symbols=["Cu", "O"],
        positions=[(0.1, 0.0, 0.0), (4.9, 0.0, 0.0)],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )
    # Direct distance is 4.8 Å (too far), MIC distance is 0.2 Å (bonded).
    assert enforce_cu_o_connectivity(atoms, CU_O_CONNECTIVITY_CUTOFF_ANG) is True


# ---------- apply_hookean_repair ----------------------------------------------


def test_repair_returns_zero_when_no_violation() -> None:
    atoms = build_bulk_cu()  # 1 atom, no pair to violate
    assert apply_hookean_repair(atoms, BoxSamplingConfig()) == 0


def test_repair_pushes_apart_too_close_cu_pair() -> None:
    atoms = Atoms(
        symbols=["Cu", "Cu"],
        positions=[(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    config = BoxSamplingConfig()
    cu_cu_min = config.min_pair_distance_ang[frozenset(["Cu", "Cu"])]
    steps = apply_hookean_repair(atoms, config)
    assert steps >= 1
    new_distance = atoms.get_distance(0, 1, mic=True)
    np.testing.assert_allclose(new_distance, cu_cu_min, atol=1e-6)


def test_repair_preserves_centre_of_mass_of_pair() -> None:
    atoms = Atoms(
        symbols=["Cu", "Cu"],
        positions=[(2.0, 0.0, 0.0), (2.5, 0.0, 0.0)],
        cell=[20.0, 20.0, 20.0],
        pbc=True,
    )
    com_before = atoms.positions.mean(axis=0)
    apply_hookean_repair(atoms, BoxSamplingConfig())
    com_after = atoms.positions.mean(axis=0)
    np.testing.assert_allclose(com_after, com_before, atol=1e-6)


def test_repair_handles_overlapping_atoms_without_dividing_by_zero() -> None:
    atoms = Atoms(
        symbols=["Cu", "O"],
        positions=[(1.0, 2.0, 3.0), (1.0, 2.0, 3.0)],  # exactly overlapping
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    # Should not blow up; should produce a non-zero separation.
    apply_hookean_repair(atoms, BoxSamplingConfig())
    d = atoms.get_distance(0, 1, mic=True)
    assert d > 0.0


def test_repair_caps_at_max_steps() -> None:
    # Construct a problem that genuinely takes >1 step: a chain of three Cu
    # atoms all too close pairwise. Each iteration only fixes the worst pair.
    atoms = Atoms(
        symbols=["Cu", "Cu", "Cu"],
        positions=[(0.0, 0.0, 0.0), (0.3, 0.0, 0.0), (0.6, 0.0, 0.0)],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    config = BoxSamplingConfig(hookean_max_steps=2)
    steps = apply_hookean_repair(atoms, config)
    assert steps <= 2


# ---------- perturb_structure -------------------------------------------------


def test_perturb_returns_fresh_atoms_when_accepted() -> None:
    rng = np.random.default_rng(42)
    seed = build_bulk_cu2o()
    result = perturb_structure(seed, BoxSamplingConfig(), rng)
    assert result.accepted
    assert result.atoms is not None
    assert result.atoms is not seed
    # Seed must not be mutated.
    assert seed.get_chemical_formula() == "Cu4O2"


def test_perturb_records_info_keys() -> None:
    rng = np.random.default_rng(0)
    result = perturb_structure(build_bulk_cu2o(), BoxSamplingConfig(), rng)
    for key in (
        "lattice_scale",
        "rattle_stdev_ang",
        "o_deleted",
        "o_inserted",
        "repair_steps",
        "cu_o_connectivity_ok",
    ):
        assert key in result.info, f"missing info key {key!r}"


def test_perturb_applies_lattice_scale_within_bounds() -> None:
    rng = np.random.default_rng(123)
    cfg = BoxSamplingConfig(
        lattice_scale=0.05, rattle_stdev_ang=0.0, max_o_insertions=0, max_o_deletions=0
    )
    seed = build_bulk_cu()
    seed_volume = seed.get_volume()
    for _ in range(20):
        result = perturb_structure(seed, cfg, rng)
        assert result.accepted
        ratio = result.atoms.get_volume() / seed_volume
        # Isotropic scaling by factor s scales volume by s**3.
        scale = result.info["lattice_scale"]
        np.testing.assert_allclose(ratio, scale**3, rtol=1e-6)
        assert 0.95**3 - 1e-9 <= ratio <= 1.05**3 + 1e-9


def test_perturb_with_zero_rattle_and_no_inserts_only_rescales() -> None:
    rng = np.random.default_rng(0)
    cfg = BoxSamplingConfig(rattle_stdev_ang=0.0, max_o_insertions=0, max_o_deletions=0)
    seed = build_bulk_cu()
    result = perturb_structure(seed, cfg, rng)
    assert result.accepted
    assert result.atoms.get_chemical_formula() == "Cu"
    # Fractional positions unchanged when no rattle and no insert/delete.
    np.testing.assert_allclose(
        result.atoms.get_scaled_positions(), seed.get_scaled_positions(), atol=1e-9
    )


def test_perturb_can_insert_and_delete_oxygens() -> None:
    rng = np.random.default_rng(7)
    cfg = BoxSamplingConfig(
        rattle_stdev_ang=0.0, lattice_scale=0.0, max_o_insertions=3, max_o_deletions=0
    )
    # Larger seed so insertions have room.
    seed = build_bulk_cu2o() * (2, 2, 2)
    result = perturb_structure(seed, cfg, rng)
    assert result.accepted
    n_o_before = sum(1 for s in seed.get_chemical_symbols() if s == "O")
    n_o_after = sum(1 for s in result.atoms.get_chemical_symbols() if s == "O")
    assert n_o_after >= n_o_before, "expected at least as many O atoms after insertion"


def test_perturb_rejects_when_connectivity_required_and_floating_o() -> None:
    rng = np.random.default_rng(0)
    cfg = BoxSamplingConfig(enforce_connectivity=True)
    # Construct a degenerate seed: a single O in a huge box, no Cu at all.
    seed = Atoms("O", positions=[(0.0, 0.0, 0.0)], cell=[20, 20, 20], pbc=True)
    result = perturb_structure(seed, cfg, rng)
    assert not result.accepted
    assert result.info["cu_o_connectivity_ok"] is False


def test_perturb_returns_disconnected_when_enforcement_off() -> None:
    rng = np.random.default_rng(0)
    cfg = BoxSamplingConfig(enforce_connectivity=False)
    seed = Atoms("O", positions=[(0.0, 0.0, 0.0)], cell=[20, 20, 20], pbc=True)
    result = perturb_structure(seed, cfg, rng)
    assert result.accepted
    assert result.info["cu_o_connectivity_ok"] is False


def test_perturb_is_reproducible_with_same_rng_seed() -> None:
    seed_atoms = build_bulk_cuo()
    cfg = BoxSamplingConfig()

    r1 = perturb_structure(seed_atoms, cfg, np.random.default_rng(42))
    r2 = perturb_structure(seed_atoms, cfg, np.random.default_rng(42))
    assert r1.accepted and r2.accepted
    np.testing.assert_allclose(r1.atoms.positions, r2.atoms.positions)
    np.testing.assert_allclose(r1.atoms.cell.array, r2.atoms.cell.array)
    assert r1.atoms.get_chemical_formula() == r2.atoms.get_chemical_formula()


def test_perturbed_structure_has_no_atoms_below_minimum_distance() -> None:
    rng = np.random.default_rng(2026)
    cfg = BoxSamplingConfig()
    seed = build_bulk_cuo() * (2, 2, 2)
    for _ in range(10):
        result = perturb_structure(seed, cfg, rng)
        if not result.accepted:
            continue
        distances = result.atoms.get_all_distances(mic=True)
        symbols = result.atoms.get_chemical_symbols()
        for i in range(len(result.atoms)):
            for j in range(i + 1, len(result.atoms)):
                r_min = cfg.min_pair_distance_ang.get(
                    frozenset([symbols[i], symbols[j]])
                )
                if r_min is None:
                    continue
                # Allow a tiny floating-point tolerance below r_min.
                assert distances[i, j] >= r_min - 1e-6, (
                    f"pair ({i},{j}) below minimum: {distances[i, j]:.3f} < {r_min}"
                )


# ---------- sample_batch ------------------------------------------------------


def test_sample_batch_zero_samples() -> None:
    rng = np.random.default_rng(0)
    results = sample_batch(build_bulk_cu(), 0, BoxSamplingConfig(), rng)
    assert results == []


def test_sample_batch_returns_requested_count() -> None:
    rng = np.random.default_rng(0)
    results = sample_batch(build_bulk_cu2o(), 5, BoxSamplingConfig(), rng)
    assert len(results) == 5
    assert all(isinstance(r, PerturbationResult) for r in results)


def test_sample_batch_retries_on_rejection() -> None:
    # Force frequent rejections by using a tiny cell where insertions and
    # connectivity conflict, then verify retry budget is respected.
    rng = np.random.default_rng(0)
    cfg = BoxSamplingConfig(max_o_insertions=0, max_o_deletions=0)
    results = sample_batch(build_bulk_cu2o(), 3, cfg, rng, max_attempts_per_sample=2)
    assert len(results) == 3
    # Each result records the number of attempts tried.
    for r in results:
        assert "attempts" in r.info
        assert 1 <= r.info["attempts"] <= 2


def test_sample_batch_rejects_invalid_arguments() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        sample_batch(build_bulk_cu(), -1, BoxSamplingConfig(), rng)
    with pytest.raises(ValueError):
        sample_batch(
            build_bulk_cu(), 1, BoxSamplingConfig(), rng, max_attempts_per_sample=0
        )


# ---------- defaults ----------------------------------------------------------


def test_default_min_pair_distances_are_reasonable() -> None:
    cfg = BoxSamplingConfig()
    # Cu-O minimum must be below typical Cu-O bond (~2.0 Å) so a
    # post-relaxation Cu-O bond at 1.9 Å is not flagged as a violation.
    assert cfg.min_pair_distance_ang[frozenset(["Cu", "O"])] < 1.9
    # Cu-O minimum must also be above absurd nuclear-collision distances.
    assert cfg.min_pair_distance_ang[frozenset(["Cu", "O"])] > 1.0


def test_default_insert_min_distance_above_cu_o_minimum() -> None:
    # Insertion is more conservative than the Hookean cutoff — we want new
    # atoms to land somewhere DFT can actually handle, not on the boundary.
    cfg = BoxSamplingConfig()
    assert (
        cfg.min_pair_distance_ang[frozenset(["Cu", "O"])]
        <= DEFAULT_INSERT_MIN_DISTANCE_ANG
    )
