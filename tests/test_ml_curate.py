"""Tests for copper_oxide_dft.ml.curate.

The SOAP/PCA/UMAP path needs ``dscribe`` + ``sklearn`` + ``umap`` (the ``ml``
extra). Those tests are skipped automatically when the deps are missing;
the rest of the module (force filter, grid subsample, train/test split,
extxyz writer) is pure ASE + numpy and is covered fully.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from copper_oxide_dft.ml.curate import (
    DEFAULT_MAX_FORCE_EV_PER_ANGSTROM,
    DEFAULT_TRAIN_RATIO,
    DatasetSplit,
    filter_by_max_force,
    prepare_dataset,
    subsample_by_grid_2d,
    train_test_split,
    write_extxyz,
)


def _atoms_with_forces(max_force: float | None) -> Atoms:
    """Make a tiny Cu atoms object with a controllable max-force value.

    A single Cu atom carrying a SinglePointCalculator with one force
    vector; ``max_force`` is the magnitude of that vector. ``None``
    leaves no calculator attached (the metadata-only path).
    """
    atoms = Atoms("Cu", positions=[(0.0, 0.0, 0.0)], cell=[10, 10, 10], pbc=True)
    if max_force is not None:
        atoms.calc = SinglePointCalculator(
            atoms, energy=0.0, forces=np.array([[max_force, 0.0, 0.0]])
        )
    return atoms


# ---------- filter_by_max_force ----------------------------------------------


def test_force_filter_keeps_below_threshold() -> None:
    items = [
        (_atoms_with_forces(1.0), {"max_force_ev_per_angstrom": 1.0}),
        (_atoms_with_forces(9.9), {"max_force_ev_per_angstrom": 9.9}),
    ]
    kept = filter_by_max_force(items, max_force_ev_per_angstrom=10.0)
    assert len(kept) == 2


def test_force_filter_drops_above_threshold() -> None:
    items = [
        (_atoms_with_forces(1.0), {"max_force_ev_per_angstrom": 1.0}),
        (_atoms_with_forces(11.0), {"max_force_ev_per_angstrom": 11.0}),
        (_atoms_with_forces(50.0), {"max_force_ev_per_angstrom": 50.0}),
    ]
    kept = filter_by_max_force(items, max_force_ev_per_angstrom=10.0)
    assert len(kept) == 1
    assert kept[0][1]["max_force_ev_per_angstrom"] == 1.0


def test_force_filter_uses_attached_calc_if_metadata_missing() -> None:
    """If metadata.max_force is absent, fall back to the structure's calculator."""
    items = [
        (_atoms_with_forces(1.0), {}),    # in-bound, no metadata
        (_atoms_with_forces(20.0), {}),   # out-of-bound, no metadata
    ]
    kept = filter_by_max_force(items, max_force_ev_per_angstrom=10.0)
    assert len(kept) == 1


def test_force_filter_drops_structure_with_no_forces() -> None:
    """No calculator and no metadata → cannot evaluate, must drop."""
    items = [(_atoms_with_forces(None), {})]
    kept = filter_by_max_force(items, max_force_ev_per_angstrom=10.0)
    assert kept == []


def test_force_filter_rejects_non_positive_threshold() -> None:
    with pytest.raises(ValueError):
        filter_by_max_force([], max_force_ev_per_angstrom=0.0)
    with pytest.raises(ValueError):
        filter_by_max_force([], max_force_ev_per_angstrom=-1.0)


def test_force_filter_preserves_order() -> None:
    items = [
        (_atoms_with_forces(1.0), {"max_force_ev_per_angstrom": 1.0, "id": "a"}),
        (_atoms_with_forces(20.0), {"max_force_ev_per_angstrom": 20.0, "id": "b"}),
        (_atoms_with_forces(2.0), {"max_force_ev_per_angstrom": 2.0, "id": "c"}),
    ]
    kept = filter_by_max_force(items, max_force_ev_per_angstrom=10.0)
    assert [meta["id"] for _, meta in kept] == ["a", "c"]


def test_default_force_threshold_matches_manuscript() -> None:
    assert DEFAULT_MAX_FORCE_EV_PER_ANGSTROM == 10.0


# ---------- subsample_by_grid_2d ---------------------------------------------


def test_grid_subsample_returns_empty_for_no_coords() -> None:
    assert subsample_by_grid_2d(np.zeros((0, 2))) == []


def test_grid_subsample_one_point_per_cell() -> None:
    # 4 points each in its own well-separated quadrant: a 2×2 grid should
    # yield exactly 4 chosen indices.
    coords = np.array([
        [0.0, 0.0], [0.1, 0.1],
        [10.0, 0.0], [10.1, 0.05],
        [0.0, 10.0], [0.05, 10.05],
        [10.0, 10.0], [10.05, 10.0],
    ])
    chosen = subsample_by_grid_2d(coords, grid_size=2, rng=np.random.default_rng(0))
    assert len(chosen) == 4
    # Every pair of chosen points should land in a different cell.
    chosen_coords = coords[chosen]
    quadrant_ids = ((chosen_coords[:, 0] > 5).astype(int) * 2
                    + (chosen_coords[:, 1] > 5).astype(int))
    assert len(set(quadrant_ids.tolist())) == 4


def test_grid_subsample_collapses_clustered_points() -> None:
    """A tight cluster of N points should collapse to one selection.

    Setup: 4 far-flung corner anchors pin the bounding box to [0, 1]×[0, 1],
    grid_size=4 gives cells of width 0.25, and a 100-point cluster placed
    well inside one cell. Only the cluster's cell should fire once.
    """
    rng = np.random.default_rng(0)
    # Place the cluster well inside the cell whose lower-left corner is
    # (0.5, 0.5) — stdev 0.01 keeps all 100 points within ±0.04 of the
    # cluster center, comfortably within the 0.25-wide cell.
    cluster = rng.normal(loc=0.625, scale=0.01, size=(100, 2))
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    coords = np.vstack([cluster, corners])
    chosen = subsample_by_grid_2d(coords, grid_size=4, rng=rng)
    # 4 corner cells + 1 cluster cell = up to 5 selections.
    assert len(chosen) <= 5
    # The cluster (first 100 indices) must have been collapsed to ≤ 1 pick.
    cluster_indices_chosen = [i for i in chosen if i < 100]
    assert len(cluster_indices_chosen) <= 1


def test_grid_subsample_handles_collinear_points() -> None:
    """All x = 0; y varies. Grid range in x is zero but the function shouldn't divide by zero."""
    coords = np.column_stack([np.zeros(10), np.linspace(0, 1, 10)])
    chosen = subsample_by_grid_2d(coords, grid_size=5)
    # 10 collinear points; with 5 bins along y at least 5 cells should be hit.
    assert 1 <= len(chosen) <= 5


def test_grid_subsample_indices_sorted_and_unique() -> None:
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 1, size=(50, 2))
    chosen = subsample_by_grid_2d(coords, grid_size=5, rng=rng)
    assert chosen == sorted(set(chosen))


def test_grid_subsample_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        subsample_by_grid_2d(np.array([1, 2, 3]))         # 1-D
    with pytest.raises(ValueError):
        subsample_by_grid_2d(np.zeros((5, 3)))            # 3-column
    with pytest.raises(ValueError):
        subsample_by_grid_2d(np.zeros((5, 2)), grid_size=0)


# ---------- train_test_split -------------------------------------------------


def test_train_test_split_empty() -> None:
    assert train_test_split([]) == ([], [])


def test_train_test_split_ten_to_one_ratio() -> None:
    items = list(range(11))
    train, test = train_test_split(items, train_ratio=10.0 / 11.0,
                                    rng=np.random.default_rng(0))
    assert len(train) == 10
    assert len(test) == 1


def test_train_test_split_no_overlap_and_full_coverage() -> None:
    items = list(range(20))
    train, test = train_test_split(items, train_ratio=0.7,
                                    rng=np.random.default_rng(7))
    assert set(train).isdisjoint(test)
    assert sorted(train + test) == items


def test_train_test_split_is_deterministic() -> None:
    items = list(range(20))
    a_train, a_test = train_test_split(items, rng=np.random.default_rng(42))
    b_train, b_test = train_test_split(items, rng=np.random.default_rng(42))
    assert a_train == b_train
    assert a_test == b_test


def test_train_test_split_rejects_bad_ratio() -> None:
    with pytest.raises(ValueError):
        train_test_split([1, 2, 3], train_ratio=0.0)
    with pytest.raises(ValueError):
        train_test_split([1, 2, 3], train_ratio=1.5)


def test_train_test_split_guarantees_at_least_one_train_item() -> None:
    """Even with a tiny ratio, at least one train item if any items at all."""
    train, test = train_test_split([1, 2], train_ratio=0.01,
                                    rng=np.random.default_rng(0))
    assert len(train) >= 1


def test_default_train_ratio_is_ten_over_eleven() -> None:
    np.testing.assert_allclose(DEFAULT_TRAIN_RATIO, 10.0 / 11.0)


# ---------- write_extxyz -----------------------------------------------------


def test_write_extxyz_creates_file_with_n_structures(tmp_path: Path) -> None:
    structs = [_atoms_with_forces(0.1), _atoms_with_forces(0.2)]
    out = write_extxyz(structs, tmp_path / "out" / "train.extxyz")
    assert out.is_file()
    # extxyz lists each frame; line 1 is atom count per frame.
    lines = out.read_text().splitlines()
    # 1-atom structures: 1 (atom count) + 1 (comment) + 1 (atom) = 3 lines per frame.
    assert len(lines) == 2 * 3


def test_write_extxyz_empty_list_writes_empty_file(tmp_path: Path) -> None:
    out = write_extxyz([], tmp_path / "empty.extxyz")
    assert out.is_file()
    assert out.read_text() == ""


# ---------- prepare_dataset (small-N fast path) ------------------------------


def test_prepare_dataset_small_input_skips_soap_path(tmp_path: Path) -> None:
    """With fewer than 4 surviving structures we skip the SOAP+UMAP stages.

    This is the path that exercises end-to-end without needing dscribe/umap
    installed.
    """
    items = [
        (_atoms_with_forces(1.0), {"max_force_ev_per_angstrom": 1.0}),
        (_atoms_with_forces(2.0), {"max_force_ev_per_angstrom": 2.0}),
    ]
    result = prepare_dataset(
        items, train_path=tmp_path / "train.extxyz", test_path=tmp_path / "test.extxyz"
    )
    assert isinstance(result, DatasetSplit)
    assert result.n_input == 2
    assert result.n_after_force_filter == 2
    assert result.n_after_subsample == 2
    assert len(result.train) + len(result.test) == 2
    assert (tmp_path / "train.extxyz").is_file()
    assert (tmp_path / "test.extxyz").is_file()


def test_prepare_dataset_force_filter_then_no_soap(tmp_path: Path) -> None:
    items = [
        (_atoms_with_forces(1.0), {"max_force_ev_per_angstrom": 1.0}),
        (_atoms_with_forces(50.0), {"max_force_ev_per_angstrom": 50.0}),
        (_atoms_with_forces(2.0), {"max_force_ev_per_angstrom": 2.0}),
    ]
    result = prepare_dataset(
        items,
        train_path=tmp_path / "train.extxyz",
        test_path=tmp_path / "test.extxyz",
        max_force_ev_per_angstrom=10.0,
    )
    assert result.n_after_force_filter == 2
    assert result.summary().startswith("input=3")


def test_prepare_dataset_when_force_filter_drops_everything(tmp_path: Path) -> None:
    """Regression: when no structures survive the filter, we must return an
    empty :class:`DatasetSplit` cleanly, not fall through into the SOAP/UMAP
    pipeline on an empty list (which would crash or silently produce garbage).
    """
    items = [
        (_atoms_with_forces(50.0), {"max_force_ev_per_angstrom": 50.0}),
        (_atoms_with_forces(99.0), {"max_force_ev_per_angstrom": 99.0}),
    ]
    result = prepare_dataset(
        items,
        train_path=tmp_path / "train.extxyz",
        test_path=tmp_path / "test.extxyz",
        max_force_ev_per_angstrom=10.0,
    )
    assert isinstance(result, DatasetSplit)
    assert result.n_input == 2
    assert result.n_after_force_filter == 0
    assert result.n_after_subsample == 0
    assert result.train == []
    assert result.test == []
    assert (tmp_path / "train.extxyz").is_file()
    assert (tmp_path / "test.extxyz").is_file()


def test_dataset_split_summary_is_human_readable(tmp_path: Path) -> None:
    items = [(_atoms_with_forces(1.0), {"max_force_ev_per_angstrom": 1.0})]
    result = prepare_dataset(
        items, train_path=tmp_path / "train.extxyz", test_path=tmp_path / "test.extxyz"
    )
    summary = result.summary()
    assert "input=" in summary
    assert "train=" in summary
    assert "test=" in summary


# ---------- SOAP/UMAP path (skipped if heavy deps missing) -------------------


def _has_ml_extras() -> bool:
    try:
        import dscribe  # noqa: F401
        import sklearn  # noqa: F401
        import umap  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_ml_extras(), reason="ML extras (dscribe/sklearn/umap) not installed")
def test_compute_soap_features_and_umap_round_trip() -> None:
    from copper_oxide_dft.ml.curate import compute_soap_features, project_to_umap_2d
    from copper_oxide_dft.structure_builder import (
        build_bulk_cu,
        build_bulk_cu2o,
        build_bulk_cuo,
    )

    structures = [build_bulk_cu(), build_bulk_cu2o(), build_bulk_cuo()]
    features = compute_soap_features(structures)
    assert features.shape[0] == 3
    assert features.ndim == 2
    coords = project_to_umap_2d(features, pca_components=2)
    assert coords.shape == (3, 2)


@pytest.mark.skipif(_has_ml_extras(), reason="Test the missing-deps contract only when deps absent")
def test_compute_soap_features_raises_importerror_without_dscribe() -> None:
    """Contract: when dscribe isn't installed, the call must raise ImportError
    *at use time*, not at module import. Lets the surrounding pipeline stay
    importable on a Mac dev environment."""
    from copper_oxide_dft.ml.curate import compute_soap_features
    from copper_oxide_dft.structure_builder import build_bulk_cu

    with pytest.raises(ImportError):
        compute_soap_features([build_bulk_cu()])


@pytest.mark.skipif(_has_ml_extras(), reason="Test the missing-deps contract only when deps absent")
def test_project_to_umap_2d_raises_importerror_without_sklearn_or_umap() -> None:
    """Same contract as above but for the sklearn + umap stage."""
    from copper_oxide_dft.ml.curate import project_to_umap_2d

    with pytest.raises(ImportError):
        project_to_umap_2d(np.zeros((4, 10)))
