"""Tests for copper_oxide_dft.ml.qe_driver."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from copper_oxide_dft.config import SystemConfig
from copper_oxide_dft.ml.qe_driver import (
    DEFAULT_FORCE_CONVERGENCE_RY_PER_BOHR,
    DEFAULT_MIXING_BETA,
    DEFAULT_SCF_CONVERGENCE_RY,
    DatasetEntry,
    read_dataset_outputs,
    write_dataset_inputs,
)
from copper_oxide_dft.structure_builder import (
    build_bulk_cu,
    build_bulk_cu2o,
)


def _assert_namelist_value(text: str, key: str, expected) -> None:
    """Assert that ``key = expected`` appears in a QE input file.

    Anchored so that a bare ``100`` elsewhere in the file (e.g., in a
    k-points line) cannot accidentally satisfy the check for ``ecutwfc = 100``.
    """
    pattern = rf"(?<![A-Za-z_]){re.escape(key)}\s*=\s*([-+0-9.eEdD]+)"
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    assert matches, f"Key {key!r} not found in input file."
    found_values = [float(m.replace("D", "e").replace("d", "e")) for m in matches]
    assert any(np.isclose(v, float(expected)) for v in found_values), (
        f"Expected {key} = {expected}, found values {found_values}"
    )


@pytest.fixture
def pseudo_dir(tmp_path: Path) -> Path:
    d = tmp_path / "pseudos"
    d.mkdir()
    (d / "Cu.upf").write_text("")
    (d / "O.upf").write_text("")
    return d


@pytest.fixture
def phase1_config() -> SystemConfig:
    """Phase 1 converged settings for bulk Cu."""
    return SystemConfig(
        ecutwfc_ry=100.0,
        kpts=(18, 18, 18),
        degauss_ry=0.01,
        extras={"lattice_a_ang": 3.6577},
    )


# ---------- write_dataset_inputs: layout & manifest ---------------------------


def test_write_dataset_creates_sample_directories(
    tmp_path: Path, pseudo_dir: Path, phase1_config: SystemConfig
) -> None:
    structures = [build_bulk_cu() for _ in range(3)]
    entries = write_dataset_inputs(
        structures, tmp_path / "ds", system_config=phase1_config, pseudo_dir=pseudo_dir
    )

    assert len(entries) == 3
    for i, entry in enumerate(entries):
        assert entry.sample_id == f"sample_{i:05d}"
        sample_dir = tmp_path / "ds" / entry.relative_path
        assert (sample_dir / "pw.in").is_file()


def test_write_dataset_writes_manifest_one_per_line(
    tmp_path: Path, pseudo_dir: Path, phase1_config: SystemConfig
) -> None:
    structures = [build_bulk_cu(), build_bulk_cu2o()]
    seed_labels = ["Cu_seed", "Cu2O_seed"]
    perts = [{"o_inserted": 0}, {"o_inserted": 1}]
    write_dataset_inputs(
        structures, tmp_path / "ds",
        system_config=phase1_config,
        seed_labels=seed_labels,
        perturbation_infos=perts,
        pseudo_dir=pseudo_dir,
    )

    lines = (tmp_path / "ds" / "manifest.jsonl").read_text().splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["seed_label"] == "Cu_seed"
    assert parsed[1]["seed_label"] == "Cu2O_seed"
    assert parsed[0]["composition"] == "Cu"
    assert parsed[1]["composition"] == "Cu4O2"
    assert parsed[0]["n_atoms"] == 1
    assert parsed[1]["n_atoms"] == 6
    assert parsed[1]["n_cu"] == 4
    assert parsed[1]["n_o"] == 2
    assert parsed[0]["perturbation"] == {"o_inserted": 0}


def test_write_dataset_appends_to_existing_manifest(
    tmp_path: Path, pseudo_dir: Path, phase1_config: SystemConfig
) -> None:
    out = tmp_path / "ds"
    write_dataset_inputs(
        [build_bulk_cu()], out, system_config=phase1_config, pseudo_dir=pseudo_dir,
        write_runner_script=False,
    )
    write_dataset_inputs(
        [build_bulk_cu2o()], out, system_config=phase1_config, pseudo_dir=pseudo_dir,
        starting_index=1, write_runner_script=False,
    )
    lines = (out / "manifest.jsonl").read_text().splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["sample_id"] == "sample_00000"
    assert parsed[1]["sample_id"] == "sample_00001"


def test_write_dataset_rejects_negative_starting_index(
    tmp_path: Path, pseudo_dir: Path, phase1_config: SystemConfig
) -> None:
    with pytest.raises(ValueError):
        write_dataset_inputs(
            [build_bulk_cu()], tmp_path / "ds",
            system_config=phase1_config, pseudo_dir=pseudo_dir, starting_index=-1,
        )


def test_write_dataset_metadata_length_mismatch_raises(
    tmp_path: Path, pseudo_dir: Path, phase1_config: SystemConfig
) -> None:
    with pytest.raises(ValueError, match="length"):
        write_dataset_inputs(
            [build_bulk_cu(), build_bulk_cu2o()], tmp_path / "ds",
            system_config=phase1_config, pseudo_dir=pseudo_dir,
            seed_labels=["only_one"],
        )


# ---------- write_dataset_inputs: pw.in content -------------------------------


def test_pw_in_for_pure_cu_has_no_spin_keywords(
    tmp_path: Path, pseudo_dir: Path, phase1_config: SystemConfig
) -> None:
    """Pure Cu doesn't need nspin=2 — only structures with O get spin+U."""
    write_dataset_inputs(
        [build_bulk_cu()], tmp_path / "ds",
        system_config=phase1_config, pseudo_dir=pseudo_dir,
    )
    text = (tmp_path / "ds" / "sample_00000" / "pw.in").read_text()
    assert "nspin" not in text.lower()
    assert "hubbard_u" not in text.lower()


def test_pw_in_for_cu_o_has_spin_and_hubbard(
    tmp_path: Path, pseudo_dir: Path, phase1_config: SystemConfig
) -> None:
    write_dataset_inputs(
        [build_bulk_cu2o()], tmp_path / "ds",
        system_config=phase1_config, pseudo_dir=pseudo_dir,
    )
    text = (tmp_path / "ds" / "sample_00000" / "pw.in").read_text()
    assert "nspin" in text.lower()
    assert "hubbard_u" in text.lower()


def test_pw_in_pins_phase1_converged_settings(
    tmp_path: Path, pseudo_dir: Path, phase1_config: SystemConfig
) -> None:
    write_dataset_inputs(
        [build_bulk_cu()], tmp_path / "ds",
        system_config=phase1_config, pseudo_dir=pseudo_dir,
    )
    text = (tmp_path / "ds" / "sample_00000" / "pw.in").read_text()
    # Phase 1 converged values, anchored to the namelist key so a coincidental
    # `100` (e.g. from electron_maxstep) cannot pass for ecutwfc.
    _assert_namelist_value(text, "ecutwfc", 100.0)
    _assert_namelist_value(text, "degauss", 0.01)


def test_pw_in_is_gamma_only(
    tmp_path: Path, pseudo_dir: Path, phase1_config: SystemConfig
) -> None:
    """Box-sampling supercells use Γ-only sampling regardless of bulk_cu kpts."""
    write_dataset_inputs(
        [build_bulk_cu2o()], tmp_path / "ds",
        system_config=phase1_config, pseudo_dir=pseudo_dir,
    )
    text = (tmp_path / "ds" / "sample_00000" / "pw.in").read_text()
    # K_POINTS automatic\n1 1 1 0 0 0 — the (1,1,1) grid is the marker.
    assert "K_POINTS" in text
    # Inspect the grid line directly: should be "1 1 1 ..." not 18 18 18.
    lines = text.splitlines()
    kpoints_line_index = next(i for i, line in enumerate(lines) if "K_POINTS" in line)
    grid_line = lines[kpoints_line_index + 1].split()
    assert grid_line[:3] == ["1", "1", "1"], (
        f"Expected Γ-only k-grid, got {grid_line[:3]}"
    )


def test_pw_in_disables_symmetry(
    tmp_path: Path, pseudo_dir: Path, phase1_config: SystemConfig
) -> None:
    write_dataset_inputs(
        [build_bulk_cu()], tmp_path / "ds",
        system_config=phase1_config, pseudo_dir=pseudo_dir,
    )
    text = (tmp_path / "ds" / "sample_00000" / "pw.in").read_text().lower()
    assert "nosym" in text
    assert "noinv" in text


def test_pw_in_carries_manuscript_tolerances(
    tmp_path: Path, pseudo_dir: Path, phase1_config: SystemConfig
) -> None:
    write_dataset_inputs(
        [build_bulk_cu()], tmp_path / "ds",
        system_config=phase1_config, pseudo_dir=pseudo_dir,
        calculation="relax",
    )
    text = (tmp_path / "ds" / "sample_00000" / "pw.in").read_text()
    _assert_namelist_value(text, "forc_conv_thr", 1.0e-3)
    _assert_namelist_value(text, "conv_thr", 1.0e-6)
    _assert_namelist_value(text, "mixing_beta", 0.3)


def test_constants_match_manuscript_walkthrough() -> None:
    """The dataset module's defaults are not arbitrary — they're load-bearing."""
    assert DEFAULT_FORCE_CONVERGENCE_RY_PER_BOHR == 1.0e-3
    assert DEFAULT_SCF_CONVERGENCE_RY == 1.0e-6
    assert DEFAULT_MIXING_BETA == 0.3


# ---------- runner script -----------------------------------------------------


def test_runner_script_skips_completed_samples(
    tmp_path: Path, pseudo_dir: Path, phase1_config: SystemConfig
) -> None:
    write_dataset_inputs(
        [build_bulk_cu(), build_bulk_cu2o()], tmp_path / "ds",
        system_config=phase1_config, pseudo_dir=pseudo_dir,
    )
    script = (tmp_path / "ds" / "run_all.sh").read_text()
    assert "qe-run" in script
    # Resume-safe pattern: don't re-run anything that's already JOB DONE.
    assert "JOB DONE" in script
    assert "sample_00000" in script and "sample_00001" in script


def test_runner_script_is_executable(
    tmp_path: Path, pseudo_dir: Path, phase1_config: SystemConfig
) -> None:
    write_dataset_inputs(
        [build_bulk_cu()], tmp_path / "ds",
        system_config=phase1_config, pseudo_dir=pseudo_dir,
    )
    script = tmp_path / "ds" / "run_all.sh"
    # Owner execute bit set.
    assert script.stat().st_mode & 0o100


def test_runner_script_not_written_when_disabled(
    tmp_path: Path, pseudo_dir: Path, phase1_config: SystemConfig
) -> None:
    write_dataset_inputs(
        [build_bulk_cu()], tmp_path / "ds",
        system_config=phase1_config, pseudo_dir=pseudo_dir,
        write_runner_script=False,
    )
    assert not (tmp_path / "ds" / "run_all.sh").exists()


# ---------- read_dataset_outputs ----------------------------------------------


def _write_fake_pw_out(path: Path, energy_ry: float = -50.0, with_job_done: bool = True) -> None:
    """Write a minimal pw.out that parse_pw_output can read.

    Note: this isn't a fully-formed QE output (it lacks the geometry block
    that ase.io.read would parse) — it's only used in the manifest-walk
    tests, not the ASE-read path.
    """
    text = (
        f"!    total energy              =  {energy_ry:.6f} Ry\n"
        + ("JOB DONE.\n" if with_job_done else "")
    )
    path.write_text(text)


def test_read_dataset_returns_empty_when_no_manifest(tmp_path: Path) -> None:
    assert read_dataset_outputs(tmp_path / "ds") == []


def test_read_dataset_skips_missing_pw_out(
    tmp_path: Path, pseudo_dir: Path, phase1_config: SystemConfig
) -> None:
    """A sample with pw.in but no pw.out is silently skipped (the run hasn't happened yet)."""
    write_dataset_inputs(
        [build_bulk_cu()], tmp_path / "ds",
        system_config=phase1_config, pseudo_dir=pseudo_dir,
    )
    # No pw.out written — read should return empty.
    assert read_dataset_outputs(tmp_path / "ds") == []


def test_read_dataset_skips_unconverged_when_required(
    tmp_path: Path, pseudo_dir: Path, phase1_config: SystemConfig
) -> None:
    write_dataset_inputs(
        [build_bulk_cu()], tmp_path / "ds",
        system_config=phase1_config, pseudo_dir=pseudo_dir,
    )
    sample = tmp_path / "ds" / "sample_00000"
    # Fake an unconverged output: total energy present, but no JOB DONE.
    _write_fake_pw_out(sample / "pw.out", with_job_done=False)
    # require_job_done=True → skipped
    assert read_dataset_outputs(tmp_path / "ds", require_job_done=True) == []


# ---------- DatasetEntry semantics --------------------------------------------


def test_dataset_entry_serializes_to_dict() -> None:
    entry = DatasetEntry(
        sample_id="sample_00007",
        relative_path="sample_00007",
        seed_label="CuO_seed",
        composition="Cu4O3",
        n_atoms=7,
        n_cu=4,
        n_o=3,
        perturbation={"o_inserted": 1, "o_deleted": 0},
    )
    payload = entry.to_json_dict()
    assert payload["sample_id"] == "sample_00007"
    assert payload["n_cu"] == 4
    assert payload["perturbation"]["o_inserted"] == 1
