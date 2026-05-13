"""Tests for copper_oxide_dft.submit."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from copper_oxide_dft.submit import (
    SlurmConfig,
    write_slurm_script,
    write_slurm_scripts_for_tree,
)


@pytest.fixture
def cfg() -> SlurmConfig:
    return SlurmConfig(
        account="CHEM000", nodes=2, ntasks_per_node=16, walltime="2:00:00"
    )


def _make_pw_in(work_dir: Path) -> None:
    work_dir.mkdir(parents=True)
    (work_dir / "pw.in").write_text("&CONTROL\n/\n")


def test_write_slurm_script_renders_expected_directives(
    tmp_path: Path, cfg: SlurmConfig
) -> None:
    _make_pw_in(tmp_path / "ecutwfc_60")
    script = write_slurm_script(tmp_path / "ecutwfc_60", cfg)

    text = script.read_text()
    assert script.name == "submit.sh"
    assert text.startswith("#!/bin/bash\n")
    assert "#SBATCH -A CHEM000" in text
    assert "#SBATCH -J ecutwfc_60" in text  # default job name = dir name
    assert "#SBATCH -p batch" in text
    assert "#SBATCH -N 2" in text
    assert "#SBATCH --ntasks-per-node=16" in text
    assert "#SBATCH -t 2:00:00" in text
    assert "module purge" in text
    assert "module load quantum-espresso" in text
    assert "srun pw.x -in pw.in > pw.out" in text


def test_write_slurm_script_marks_executable(tmp_path: Path, cfg: SlurmConfig) -> None:
    _make_pw_in(tmp_path / "run")
    script = write_slurm_script(tmp_path / "run", cfg)
    mode = stat.S_IMODE(script.stat().st_mode)
    assert mode & stat.S_IXUSR


def test_write_slurm_script_custom_job_name(tmp_path: Path, cfg: SlurmConfig) -> None:
    _make_pw_in(tmp_path / "run")
    script = write_slurm_script(tmp_path / "run", cfg, job_name="bulk_cu_validation")
    assert "#SBATCH -J bulk_cu_validation" in script.read_text()


def test_write_slurm_script_requires_pw_in(tmp_path: Path, cfg: SlurmConfig) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="pw.in"):
        write_slurm_script(tmp_path / "empty", cfg)


def test_write_slurm_script_extra_modules_and_exports(tmp_path: Path) -> None:
    _make_pw_in(tmp_path / "run")
    cfg = SlurmConfig(
        account="CHEM000",
        extra_modules=("hdf5/1.14",),
        extra_exports={"OMP_NUM_THREADS": "1"},
    )
    text = write_slurm_script(tmp_path / "run", cfg).read_text()
    assert "module load hdf5/1.14" in text
    assert 'export OMP_NUM_THREADS="1"' in text


def test_write_slurm_scripts_for_tree(tmp_path: Path, cfg: SlurmConfig) -> None:
    for name in ("ecutwfc_40", "ecutwfc_60", "ecutwfc_80"):
        _make_pw_in(tmp_path / "conv" / name)

    scripts = write_slurm_scripts_for_tree(tmp_path / "conv", cfg)

    assert len(scripts) == 3
    for s in scripts:
        assert s.name == "submit.sh"
        assert (s.parent / "pw.in").is_file()
        # Job name should match the directory containing pw.in
        assert f"#SBATCH -J {s.parent.name}" in s.read_text()


def test_write_slurm_scripts_for_tree_empty(tmp_path: Path, cfg: SlurmConfig) -> None:
    (tmp_path / "empty").mkdir()
    assert write_slurm_scripts_for_tree(tmp_path / "empty", cfg) == []
