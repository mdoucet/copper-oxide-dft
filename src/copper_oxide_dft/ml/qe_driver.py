"""Batched QE input generation + output collection for the MLIP-GCGO dataset.

Given a list of perturbed structures from :mod:`copper_oxide_dft.ml.box_sampling`,
this module:

1. Writes one ``pw.in`` per structure into a per-sample directory.
2. Emits a JSON-lines manifest (``manifest.jsonl``) capturing seed label,
   composition, atom count, and the perturbation metadata.
3. Optionally writes a ``run_all.sh`` driver that loops a single-rank
   ``qe-run`` over each sample directory (matches the workflow in
   :doc:`/docs/startup-cuo-cu-nonaqueous.md`).
4. After the user has executed the QE runs, reads ``pw.out`` files back
   into :class:`ase.Atoms` with energy + forces attached, ready for the
   :mod:`copper_oxide_dft.ml.curate` filtering/subsampling step.

The actual ``pw.x`` execution is **not** invoked here. On the DGX Spark
workstation we keep the same explicit ``qe-run <dir>`` loop the rest of
the project uses: one command per sample, easy to inspect, restartable
on failure. Embedding `subprocess.run("pw.x", ...)` here would couple
input generation to execution and obscure where time and GPU were spent.

Design choices that the manuscript walkthrough pins:

- **Γ-only k-points**: the 100+ atom perturbed supercells are too big
  for a finite k-grid to be worth the cost. Inherited from
  :doc:`/docs/machine-learned-dft.md` §2.
- **``nosym=True`` / ``noinv=True``**: random perturbations break the
  seed cells' space groups, so QE's automatic symmetry detection would
  pick the wrong (over-symmetric) k-set. Same rationale as the
  constrained-slab case in
  :doc:`/docs/ground_truths.md`: "Slab relaxations need nosym=True".
- **Tight tolerances**: ``forc_conv_thr = 1.0e-3 Ry/Bohr``,
  ``conv_thr = 1.0e-6 Ry``. These match the manuscript's bulk
  box-sampling and ensure the MLIP fine-tune sees consistent labels.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ase import Atoms

from copper_oxide_dft.config import SystemConfig
from copper_oxide_dft.qe_input import (
    DEFAULT_HUBBARD_U_CU_3D_EV,
    DEFAULT_PSEUDOPOTENTIALS,
    merge_namelist_overrides,
    write_pw_input,
)

DEFAULT_FORCE_CONVERGENCE_RY_PER_BOHR = 1.0e-3
"""Force convergence threshold for relax/vc-relax. ~2.5e-2 eV/Å.
Manuscript :doc:`/docs/machine-learned-dft.md` value; tight enough that
MACE labels are noise-free at the ~10 meV/Å test-MAE we're targeting."""

DEFAULT_SCF_CONVERGENCE_RY = 1.0e-6
"""SCF convergence threshold (~10 µeV). Slightly looser than the
production-quality ``write_pw_input`` default (1e-8) to keep the
~5000-structure dataset tractable on DGX Spark."""

DEFAULT_MIXING_BETA = 0.3
"""Aggressive dampening for highly-perturbed starting guesses. The
manuscript uses 0.3; ``write_pw_input``'s default of 0.4 is fine for
near-equilibrium geometries but harder to converge for box-sampling."""

DEFAULT_ELECTRON_MAXSTEP = 100
"""Max SCF iterations. Beyond ~100 the structure is probably pathological
and worth dropping rather than burning more wall time on."""

DEFAULT_BOX_SAMPLING_KPTS: tuple[int, int, int] = (1, 1, 1)
"""Γ-only sampling is the right call for ~100+ atom box-sampling cells —
a finite grid would be unaffordable. Callers building smaller cells can
override; see :doc:`/docs/machine-learned-dft.md` §2."""


@dataclass(frozen=True)
class DatasetEntry:
    """One row of the dataset manifest.

    Captures everything needed to (1) re-run a sample if QE failed, and
    (2) attribute a downstream MACE training error back to the
    perturbation that produced it.
    """

    sample_id: str
    relative_path: str
    seed_label: str
    composition: str
    n_atoms: int
    n_cu: int
    n_o: int
    perturbation: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_dataset_inputs(
    structures: Iterable[Atoms],
    out_root: str | os.PathLike[str],
    *,
    system_config: SystemConfig,
    seed_labels: Iterable[str] | None = None,
    perturbation_infos: Iterable[Mapping[str, Any]] | None = None,
    pseudopotentials: Mapping[str, str] | None = None,
    calculation: str = "relax",
    hubbard_u_ev: float = DEFAULT_HUBBARD_U_CU_3D_EV,
    pseudo_dir: str | os.PathLike[str] | None = None,
    kpts: tuple[int, int, int] = DEFAULT_BOX_SAMPLING_KPTS,
    write_runner_script: bool = True,
    starting_index: int = 0,
) -> list[DatasetEntry]:
    """Write per-sample ``pw.in`` files + a manifest for a batch of structures.

    Args:
        structures: Iterable of ASE :class:`Atoms` to write inputs for.
            Caller is responsible for providing already-perturbed,
            connectivity-checked structures (see
            :func:`copper_oxide_dft.ml.box_sampling.sample_batch`).
        out_root: Directory to populate. Created if missing.
            Layout: ``out_root/sample_00000/pw.in``,
            ``out_root/sample_00001/pw.in``, ...,
            ``out_root/manifest.jsonl``,
            ``out_root/run_all.sh`` *(optional)*.
        system_config: Phase 1 :class:`SystemConfig` (typically
            ``bulk_cu``). Provides ``ecutwfc_ry`` and ``degauss_ry``.
            ``kpts`` is overridden to Γ-only — perturbed supercells are
            large enough that a finite grid would be unaffordable.
        seed_labels: Optional per-structure labels (e.g. ``"Cu2O_seed"``)
            written into the manifest. Defaults to ``"unknown"``.
        perturbation_infos: Optional per-structure dicts from
            :class:`copper_oxide_dft.ml.box_sampling.PerturbationResult.info`.
        pseudopotentials: Mapping species → UPF filename.
        calculation: ``"relax"`` (default; positions only) or ``"vc-relax"``
            (cell + positions; matches manuscript for bulk box-sampling).
        hubbard_u_ev: Hubbard U on Cu 3d. Default matches Phase 4 / 7.
        pseudo_dir: Directory containing the UPF files. Falls back to
            ``$CUOXDFT_PSEUDO_DIR``.
        write_runner_script: Emit ``run_all.sh`` that loops ``qe-run`` over
            each sample directory.
        starting_index: Sample numbering offset. Use a non-zero value to
            append to an existing manifest without filename collisions.

    Returns:
        List of :class:`DatasetEntry` describing the samples written.

    Raises:
        ValueError: If ``starting_index < 0`` or if the iterables passed
            for ``seed_labels`` / ``perturbation_infos`` have a different
            length than ``structures``.
    """
    if starting_index < 0:
        raise ValueError(f"starting_index must be >= 0; got {starting_index}.")
    if pseudopotentials is None:
        pseudopotentials = DEFAULT_PSEUDOPOTENTIALS

    structures_list = list(structures)
    seed_labels_list = _zip_or_default(seed_labels, structures_list, default="unknown")
    perturbation_list = _zip_or_default(perturbation_infos, structures_list, default={})

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    entries: list[DatasetEntry] = []
    for offset, (atoms, seed_label, pert_info) in enumerate(
        zip(structures_list, seed_labels_list, perturbation_list, strict=True)
    ):
        sample_id = f"sample_{starting_index + offset:05d}"
        sample_dir = out_root / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        extra_input_data, additional_cards = _build_qe_input_pieces(
            atoms, hubbard_u_ev
        )

        write_pw_input(
            atoms,
            out_path=sample_dir / "pw.in",
            pseudopotentials=pseudopotentials,
            calculation=calculation,
            prefix=sample_id,
            ecutwfc=system_config.ecutwfc_ry,
            kpts=kpts,
            degauss=system_config.degauss_ry,
            pseudo_dir=pseudo_dir,
            extra_input_data=extra_input_data,
            additional_cards=additional_cards,
        )

        entry = DatasetEntry(
            sample_id=sample_id,
            relative_path=sample_id,
            seed_label=seed_label,
            composition=atoms.get_chemical_formula(),
            n_atoms=len(atoms),
            n_cu=sum(1 for s in atoms.get_chemical_symbols() if s == "Cu"),
            n_o=sum(1 for s in atoms.get_chemical_symbols() if s == "O"),
            perturbation=dict(pert_info),
        )
        entries.append(entry)

    _append_manifest(out_root / "manifest.jsonl", entries)

    if write_runner_script:
        _write_runner_script(out_root / "run_all.sh", entries)

    return entries


def read_dataset_outputs(
    out_root: str | os.PathLike[str],
    *,
    require_job_done: bool = True,
) -> list[tuple[Atoms, dict[str, Any]]]:
    """Load all completed QE outputs for a dataset directory.

    Walks the manifest at ``out_root/manifest.jsonl`` and, for each entry
    that has a ``pw.out`` next to it, reads the final geometry, energy,
    and forces via ASE's QE output parser. Failed or missing runs are
    silently skipped (their absence shows up in the returned count vs the
    manifest size — callers can compute the success rate).

    Args:
        out_root: Directory previously populated by
            :func:`write_dataset_inputs`.
        require_job_done: If True, structures without ``JOB DONE.`` in
            ``pw.out`` are skipped (matches the manuscript's "remove
            unconverged frames" step). Set False only for triage.

    Returns:
        List of ``(atoms, metadata)`` tuples. ``atoms`` has energy and
        forces attached via ASE's :class:`SinglePointCalculator`.
        ``metadata`` is the manifest entry as a dict, with added keys
        ``max_force_ev_per_angstrom`` and ``job_done``.
    """
    out_root = Path(out_root)
    manifest_path = out_root / "manifest.jsonl"
    if not manifest_path.is_file():
        return []

    from ase.io import read as ase_read

    from copper_oxide_dft.parse import parse_pw_output

    results: list[tuple[Atoms, dict[str, Any]]] = []
    for entry_dict in _iter_manifest(manifest_path):
        pw_out = out_root / entry_dict["relative_path"] / "pw.out"
        if not pw_out.is_file():
            continue

        try:
            scalars = parse_pw_output(pw_out)
        except ValueError:
            # No converged total energy line at all.
            continue

        if require_job_done and not scalars.job_done:
            continue

        try:
            atoms = ase_read(pw_out, format="espresso-out")
        except Exception:  # noqa: BLE001 — ASE raises a grab-bag of issues
            continue

        forces = atoms.get_forces() if atoms.calc is not None else None
        max_force = (
            float((forces ** 2).sum(axis=1).max() ** 0.5)
            if forces is not None and len(forces) > 0
            else None
        )

        metadata = {
            **entry_dict,
            "job_done": scalars.job_done,
            "max_force_ev_per_angstrom": max_force,
            "total_energy_ev": scalars.total_energy_ev,
        }
        results.append((atoms, metadata))

    return results


def _build_qe_input_pieces(
    atoms: Atoms, hubbard_u_ev: float
) -> tuple[dict[str, dict[str, Any]], str]:
    """Assemble the per-sample namelist overrides + HUBBARD card.

    Combines: symmetry off, tight tolerances, aggressive mixing, spin +
    Hubbard U when O is present. Mirrors the manuscript's bulk-sampling
    recipe and the project's slab-relaxation convention. Returns the
    pieces separately because QE 7.1+ moved Hubbard U out of the
    ``&SYSTEM`` namelist into a dedicated ``HUBBARD`` card.
    """
    base = {
        "control": {"forc_conv_thr": DEFAULT_FORCE_CONVERGENCE_RY_PER_BOHR},
        "system": {"nosym": True, "noinv": True},
        "electrons": {
            "conv_thr": DEFAULT_SCF_CONVERGENCE_RY,
            "mixing_beta": DEFAULT_MIXING_BETA,
            "electron_maxstep": DEFAULT_ELECTRON_MAXSTEP,
        },
    }
    hubbard_card = ""

    if any(s == "O" for s in atoms.get_chemical_symbols()):
        from copper_oxide_dft.qe_input import spin_and_hubbard_overrides

        spin = spin_and_hubbard_overrides(
            atoms, nspin=2, hubbard_u={"Cu": hubbard_u_ev}
        )
        base = merge_namelist_overrides(base, spin.namelist_overrides)
        hubbard_card = spin.hubbard_card

    return base, hubbard_card


def _zip_or_default(values, structures, *, default):
    if values is None:
        return [default] * len(structures)
    values_list = list(values)
    if len(values_list) != len(structures):
        raise ValueError(
            f"Per-structure metadata length ({len(values_list)}) does not match "
            f"number of structures ({len(structures)})."
        )
    return values_list


def _append_manifest(manifest_path: Path, entries: list[DatasetEntry]) -> None:
    """Append entries as JSON-lines. Creates the file if absent."""
    with manifest_path.open("a") as fh:
        for entry in entries:
            fh.write(json.dumps(entry.to_json_dict(), sort_keys=True) + "\n")


def _iter_manifest(manifest_path: Path):
    with manifest_path.open() as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_runner_script(script_path: Path, entries: list[DatasetEntry]) -> None:
    """Emit a deterministic ``run_all.sh`` that calls ``qe-run`` on each sample.

    The wrapper itself lives in ``~/bin/qe-run`` per
    :doc:`/docs/startup-cuo-cu-nonaqueous.md` §1.6 — it handles the
    ``mpirun -n 1 pw.x`` invocation and OpenMP thread count.
    """
    lines = [
        "#!/usr/bin/env bash",
        "# Auto-generated by copper_oxide_dft.ml.qe_driver.write_dataset_inputs.",
        "# Runs qe-run sequentially across each sample. Resume-safe: samples",
        "# that already have a pw.out with JOB DONE are skipped.",
        "set -euo pipefail",
        "here=\"$(cd \"$(dirname \"$0\")\" && pwd)\"",
        "",
    ]
    for entry in entries:
        # Use bash-conditional skip so partial restart is cheap.
        lines.append(
            f"if ! grep -q 'JOB DONE' \"$here/{entry.relative_path}/pw.out\" 2>/dev/null; then\n"
            f"    qe-run \"$here/{entry.relative_path}\"\n"
            "fi"
        )
    lines.append("")
    script_path.write_text("\n".join(lines))
    script_path.chmod(0o755)
