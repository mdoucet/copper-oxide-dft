"""Curate DFT box-sampling outputs into a MACE training set.

Pipeline (see :doc:`/docs/machine-learned-dft.md` §3):

1. **Force filter** — drop structures whose max atomic force exceeds
   ``max_force_ev_per_angstrom`` (default 10 eV/Å, manuscript value).
   Catches pathological geometries the SCF converged on by accident.
2. **SOAP descriptors** — featurize each surviving structure into a
   global-average SOAP vector.
3. **Incremental PCA → 50 components** — compress the SOAP space.
4. **UMAP → 2-D** — project the PCA space for visual coverage analysis.
5. **20×20 grid subsampling** — pick one structure per occupied UMAP
   cell to remove physical redundancies without throwing away diversity.
6. **10:1 train/test split** — manuscript ratio.
7. **Extended-XYZ writer** — `cuox_train.extxyz` / `cuox_test.extxyz`,
   the file format ``mace_run_train`` consumes.

Pure-ASE / pure-numpy stages (1, 5–7) are testable without the heavy
ML stack. Stages 2–4 lazily import :mod:`dscribe`, :mod:`sklearn`, and
:mod:`umap` — install via ``pip install -e ".[ml]"`` on the DGX Spark.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

DEFAULT_MAX_FORCE_EV_PER_ANGSTROM = 10.0
"""Force-filter threshold from :doc:`/docs/machine-learned-dft.md`. Any
structure with a single atom whose force magnitude exceeds this is dropped
as a likely unphysical artefact."""

DEFAULT_TRAIN_RATIO = 10.0 / 11.0
"""Manuscript 10:1 train/test split."""

DEFAULT_GRID_SIZE = 20
"""20×20 grid over the UMAP plane — manuscript default."""

DEFAULT_SOAP_PARAMS: dict[str, Any] = {
    "species": ["Cu", "O"],
    "r_cut": 5.0,
    "n_max": 6,
    "l_max": 6,
    "periodic": True,
    "average": "inner",
}
"""SOAP knobs that work on Cu-O bulks. r_cut=5 Å captures second-shell
geometry without blowing up the descriptor dimension. ``average="inner"``
gives one vector per structure (global SOAP)."""


@dataclass(frozen=True)
class DatasetSplit:
    """Results of :func:`prepare_dataset` so callers can inspect what happened."""

    train: list[Atoms]
    test: list[Atoms]
    n_input: int
    n_after_force_filter: int
    n_after_subsample: int

    def summary(self) -> str:
        return (
            f"input={self.n_input}  "
            f"after_force_filter={self.n_after_force_filter}  "
            f"after_subsample={self.n_after_subsample}  "
            f"train={len(self.train)}  test={len(self.test)}"
        )


# ---------- Stage 1: force filter --------------------------------------------


def filter_by_max_force(
    items: Iterable[tuple[Atoms, dict[str, Any]]],
    max_force_ev_per_angstrom: float = DEFAULT_MAX_FORCE_EV_PER_ANGSTROM,
) -> list[tuple[Atoms, dict[str, Any]]]:
    """Drop structures whose ``max_force_ev_per_angstrom`` exceeds threshold.

    Reads the per-structure force magnitude from ``metadata`` (set by
    :func:`copper_oxide_dft.ml.qe_driver.read_dataset_outputs`). If the
    field is missing or None, recomputes it from the attached calculator;
    if no forces are available at all, the structure is dropped.

    Args:
        items: ``(atoms, metadata)`` pairs from
            :func:`read_dataset_outputs`.
        max_force_ev_per_angstrom: Upper bound on the per-atom force.

    Returns:
        Subset of ``items`` that pass the filter, preserving order.
    """
    if max_force_ev_per_angstrom <= 0:
        raise ValueError(
            f"max_force_ev_per_angstrom must be positive; got {max_force_ev_per_angstrom}."
        )

    kept: list[tuple[Atoms, dict[str, Any]]] = []
    for atoms, metadata in items:
        max_force = metadata.get("max_force_ev_per_angstrom")
        if max_force is None:
            max_force = _safe_compute_max_force(atoms)
        if max_force is None:
            continue
        if max_force <= max_force_ev_per_angstrom:
            kept.append((atoms, metadata))
    return kept


def _safe_compute_max_force(atoms: Atoms) -> float | None:
    if atoms.calc is None:
        return None
    try:
        forces = atoms.get_forces()
    except Exception:  # noqa: BLE001 — ASE forces can fail many ways
        return None
    if forces is None or len(forces) == 0:
        return None
    return float(np.linalg.norm(forces, axis=1).max())


# ---------- Stages 2-4: SOAP + PCA + UMAP (lazy imports) ---------------------


def compute_soap_features(
    structures: Sequence[Atoms],
    soap_params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Featurize structures with SOAP. Lazy-imports :mod:`dscribe`.

    Args:
        structures: ASE structures (any size, any composition).
        soap_params: Keyword args forwarded to :class:`dscribe.descriptors.SOAP`.
            Defaults to :data:`DEFAULT_SOAP_PARAMS` (Cu/O species, r_cut=5 Å,
            global-average inner product).

    Returns:
        ``(n_structures, n_soap_features)`` array of descriptors.
    """
    from dscribe.descriptors import SOAP  # lazy: heavy import

    params = dict(soap_params) if soap_params is not None else dict(DEFAULT_SOAP_PARAMS)
    soap = SOAP(**params)
    return np.asarray(soap.create(list(structures)))


def project_to_umap_2d(
    features: np.ndarray,
    *,
    pca_components: int = 50,
    random_state: int = 0,
) -> np.ndarray:
    """Compress SOAP features via Incremental PCA then project to 2-D UMAP.

    Mirrors :doc:`/docs/machine-learned-dft.md` §3 step 2. The IPCA step
    is what keeps the UMAP fit memory-bounded for ~10⁴-structure datasets.

    Args:
        features: ``(n_samples, n_features)`` SOAP array.
        pca_components: Manuscript value is 50.
        random_state: UMAP/PCA seed for reproducibility.

    Returns:
        ``(n_samples, 2)`` UMAP coordinates.
    """
    if features.shape[0] == 0:
        return np.zeros((0, 2))

    import umap  # lazy
    from sklearn.decomposition import IncrementalPCA  # lazy

    n_components = min(pca_components, features.shape[1], features.shape[0])
    ipca = IncrementalPCA(n_components=n_components)
    pca_features = ipca.fit_transform(features)

    reducer = umap.UMAP(n_components=2, random_state=random_state)
    return np.asarray(reducer.fit_transform(pca_features))


# ---------- Stage 5: grid subsampling (pure numpy) ---------------------------


def subsample_by_grid_2d(
    coords_2d: np.ndarray,
    grid_size: int = DEFAULT_GRID_SIZE,
    rng: np.random.Generator | None = None,
) -> list[int]:
    """Pick one index per occupied grid cell over the bounding box of ``coords_2d``.

    Removes physical redundancies (clusters of near-identical structures in
    descriptor space) without throwing away diversity. Mirrors the
    manuscript's "20×20 grid" subsampling step.

    Args:
        coords_2d: ``(n, 2)`` array of 2-D coordinates (typically UMAP).
        grid_size: Number of bins along each axis. The grid covers the
            bounding box of ``coords_2d``.
        rng: Generator for the per-cell pick. Pass an explicit one for
            reproducibility (the order of points within a cell is otherwise
            input-dependent).

    Returns:
        List of selected indices into ``coords_2d``, sorted ascending.

    Raises:
        ValueError: If ``grid_size`` < 1 or ``coords_2d`` is not 2-D.
    """
    if grid_size < 1:
        raise ValueError(f"grid_size must be >= 1; got {grid_size}.")
    if coords_2d.ndim != 2 or coords_2d.shape[1] != 2:
        raise ValueError(f"coords_2d must be shape (n, 2); got {coords_2d.shape}.")
    if coords_2d.shape[0] == 0:
        return []

    rng = rng or np.random.default_rng(0)

    x_min, y_min = coords_2d.min(axis=0)
    x_max, y_max = coords_2d.max(axis=0)
    # max(., 1e-12) guards against collinear inputs (all x equal); the
    # clip below catches the max-point edge case so we don't need an
    # epsilon-shrunk range here.
    x_range = max(x_max - x_min, 1e-12)
    y_range = max(y_max - y_min, 1e-12)

    ix = np.clip(((coords_2d[:, 0] - x_min) / x_range * grid_size).astype(int),
                  0, grid_size - 1)
    iy = np.clip(((coords_2d[:, 1] - y_min) / y_range * grid_size).astype(int),
                  0, grid_size - 1)
    cell_ids = ix * grid_size + iy

    chosen: list[int] = []
    for cell in np.unique(cell_ids):
        in_cell = np.where(cell_ids == cell)[0]
        chosen.append(int(rng.choice(in_cell)))
    return sorted(chosen)


# ---------- Stage 6: train/test split ----------------------------------------


def train_test_split(
    items: Sequence[Any],
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    rng: np.random.Generator | None = None,
) -> tuple[list[Any], list[Any]]:
    """Randomly partition ``items`` into (train, test) at the given ratio.

    ``train_ratio`` is interpreted as ``len(train) / len(items)``. The
    manuscript's 10:1 split corresponds to ``train_ratio = 10/11`` (test
    is 1/11). Returns shuffled lists.

    Args:
        items: Sequence to split (typically ASE Atoms).
        train_ratio: Fraction in the train set; must be in (0, 1].
        rng: Generator (pass an explicit one for reproducibility).

    Returns:
        ``(train, test)`` tuple of lists.
    """
    if not 0.0 < train_ratio <= 1.0:
        raise ValueError(f"train_ratio must be in (0, 1]; got {train_ratio}.")
    if len(items) == 0:
        return [], []

    rng = rng or np.random.default_rng(0)
    indices = list(range(len(items)))
    rng.shuffle(indices)
    split = int(round(len(items) * train_ratio))
    # Guarantee at least one train item if any items at all.
    split = max(1, min(split, len(items)))
    train = [items[i] for i in indices[:split]]
    test = [items[i] for i in indices[split:]]
    return train, test


# ---------- Stage 7: extxyz writer -------------------------------------------


def write_extxyz(
    structures: Sequence[Atoms],
    out_path: str | os.PathLike[str],
) -> Path:
    """Write ``structures`` to an Extended-XYZ file consumable by MACE.

    Args:
        structures: ASE structures with energy + forces attached
            (typically via ``ase.io.read(pw.out, format="espresso-out")``,
            which puts a :class:`SinglePointCalculator` on each).
        out_path: Destination ``.extxyz`` path.

    Returns:
        Resolved Path of the written file.
    """
    from ase.io import write as ase_write

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ase_write(str(out_path), list(structures), format="extxyz")
    return out_path


# ---------- End-to-end orchestrator ------------------------------------------


def prepare_dataset(
    items: Iterable[tuple[Atoms, dict[str, Any]]],
    train_path: str | os.PathLike[str],
    test_path: str | os.PathLike[str],
    *,
    max_force_ev_per_angstrom: float = DEFAULT_MAX_FORCE_EV_PER_ANGSTROM,
    grid_size: int = DEFAULT_GRID_SIZE,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    soap_params: dict[str, Any] | None = None,
    pca_components: int = 50,
    rng_seed: int = 0,
) -> DatasetSplit:
    """Run the full curation pipeline end-to-end.

    Args:
        items: ``(atoms, metadata)`` pairs from
            :func:`copper_oxide_dft.ml.qe_driver.read_dataset_outputs`.
        train_path: Output ``.extxyz`` for the training split.
        test_path: Output ``.extxyz`` for the test split.
        max_force_ev_per_angstrom: Stage-1 filter threshold.
        grid_size: Stage-5 UMAP grid resolution.
        train_ratio: Stage-6 train fraction (manuscript: 10/11).
        soap_params: Stage-2 SOAP knobs; defaults to
            :data:`DEFAULT_SOAP_PARAMS`.
        pca_components: Stage-3 IPCA dimension.
        rng_seed: Seed for both UMAP and the train/test shuffle.

    Returns:
        :class:`DatasetSplit` with counts at each pipeline stage and the
        train/test structures actually written.

    Raises:
        ImportError: If ``dscribe``, ``sklearn``, or ``umap`` are not
            installed (lazy-imported by the SOAP / IPCA / UMAP stages).
    """
    items_list = list(items)
    n_input = len(items_list)

    filtered = filter_by_max_force(items_list, max_force_ev_per_angstrom)
    n_after_force_filter = len(filtered)
    if n_after_force_filter == 0:
        # Write empty extxyz files so callers still see the expected outputs.
        write_extxyz([], train_path)
        write_extxyz([], test_path)
        return DatasetSplit([], [], n_input, 0, 0)

    atoms_only = [atoms for atoms, _ in filtered]

    rng = np.random.default_rng(rng_seed)

    if len(atoms_only) >= 4:
        soap_features = compute_soap_features(atoms_only, soap_params=soap_params)
        umap_coords = project_to_umap_2d(
            soap_features, pca_components=pca_components, random_state=rng_seed
        )
        selected_indices = subsample_by_grid_2d(umap_coords, grid_size=grid_size, rng=rng)
        subsampled = [atoms_only[i] for i in selected_indices]
    else:
        # Below the SOAP/UMAP threshold (typically only happens in tiny
        # sanity runs): pass everything through.
        subsampled = atoms_only

    n_after_subsample = len(subsampled)

    train, test = train_test_split(subsampled, train_ratio=train_ratio, rng=rng)

    write_extxyz(train, train_path)
    write_extxyz(test, test_path)

    return DatasetSplit(
        train=train,
        test=test,
        n_input=n_input,
        n_after_force_filter=n_after_force_filter,
        n_after_subsample=n_after_subsample,
    )
