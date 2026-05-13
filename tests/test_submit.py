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


def _make_pw_in(work_dir: Path) -> None:
    work_dir.mkdir(parents=True)
    (work_dir / "pw.in").write_text("&CONTROL\n/\n")


# ---- CPU / Andes-style ------------------------------------------------------


@pytest.fixture
def andes_cfg() -> SlurmConfig:
    return SlurmConfig.for_andes("CHM999", nodes=2, walltime="2:00:00")


def test_andes_script_renders_cpu_directives(
    tmp_path: Path, andes_cfg: SlurmConfig
) -> None:
    _make_pw_in(tmp_path / "ecutwfc_60")
    script = write_slurm_script(tmp_path / "ecutwfc_60", andes_cfg)

    text = script.read_text()
    assert text.startswith("#!/bin/bash\n")
    assert "#SBATCH -A CHM999" in text
    assert "#SBATCH -J ecutwfc_60" in text
    assert "#SBATCH -p batch" in text
    assert "#SBATCH -N 2" in text
    assert "#SBATCH --ntasks-per-node=32" in text
    assert "#SBATCH -t 2:00:00" in text
    # Andes preset: no GPU directives
    assert "--gpus-per-node" not in text
    assert "--gpu-bind" not in text
    assert "MPICH_GPU_SUPPORT_ENABLED" not in text
    assert "module load quantum-espresso" in text
    assert "srun pw.x -in pw.in > pw.out" in text


def test_script_marks_executable(tmp_path: Path, andes_cfg: SlurmConfig) -> None:
    _make_pw_in(tmp_path / "run")
    script = write_slurm_script(tmp_path / "run", andes_cfg)
    assert stat.S_IMODE(script.stat().st_mode) & stat.S_IXUSR


def test_custom_job_name(tmp_path: Path, andes_cfg: SlurmConfig) -> None:
    _make_pw_in(tmp_path / "run")
    script = write_slurm_script(tmp_path / "run", andes_cfg, job_name="custom_name")
    assert "#SBATCH -J custom_name" in script.read_text()


def test_missing_pw_in_raises(tmp_path: Path, andes_cfg: SlurmConfig) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="pw.in"):
        write_slurm_script(tmp_path / "empty", andes_cfg)


# ---- GPU / Frontier ---------------------------------------------------------


@pytest.fixture
def frontier_cfg() -> SlurmConfig:
    return SlurmConfig.for_frontier("CHM999")


def test_frontier_preset_field_defaults(frontier_cfg: SlurmConfig) -> None:
    assert frontier_cfg.partition == "batch"
    assert frontier_cfg.ntasks_per_node == 8  # 1 per GCD
    assert frontier_cfg.cpus_per_task == 7
    assert frontier_cfg.gpus_per_node == 8
    assert frontier_cfg.gpu_bind == "closest"
    assert frontier_cfg.gpu_aware_mpi is True
    assert "PrgEnv-gnu" in frontier_cfg.extra_modules
    assert "rocm" in frontier_cfg.extra_modules


def test_frontier_script_renders_gpu_directives(
    tmp_path: Path, frontier_cfg: SlurmConfig
) -> None:
    _make_pw_in(tmp_path / "run")
    text = write_slurm_script(tmp_path / "run", frontier_cfg).read_text()

    assert "#SBATCH --ntasks-per-node=8" in text
    assert "#SBATCH -c 7" in text
    assert "#SBATCH --gpus-per-node=8" in text
    assert 'export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"' in text
    assert 'export MPICH_GPU_SUPPORT_ENABLED="1"' in text
    assert "module load quantum-espresso" in text
    assert "module load PrgEnv-gnu" in text
    assert "module load rocm" in text
    assert "srun --gpus-per-task=1 --gpu-bind=closest pw.x" in text


def test_frontier_overrides_apply(tmp_path: Path) -> None:
    cfg = SlurmConfig.for_frontier("CHM999", nodes=4, walltime="6:00:00")
    _make_pw_in(tmp_path / "run")
    text = write_slurm_script(tmp_path / "run", cfg).read_text()
    assert "#SBATCH -N 4" in text
    assert "#SBATCH -t 6:00:00" in text
    # Override should preserve GPU settings.
    assert "#SBATCH --gpus-per-node=8" in text


# ---- Custom configurations --------------------------------------------------


def test_custom_extras(tmp_path: Path) -> None:
    cfg = SlurmConfig(
        account="CHM999",
        extra_modules=("hdf5/1.14",),
        extra_exports={"OMP_NUM_THREADS": "1"},
    )
    _make_pw_in(tmp_path / "run")
    text = write_slurm_script(tmp_path / "run", cfg).read_text()
    assert "module load hdf5/1.14" in text
    assert 'export OMP_NUM_THREADS="1"' in text


# ---- Sweep tree -------------------------------------------------------------


def test_tree_emits_one_script_per_input(
    tmp_path: Path, frontier_cfg: SlurmConfig
) -> None:
    for name in ("ecutwfc_40", "ecutwfc_60", "ecutwfc_80"):
        _make_pw_in(tmp_path / "conv" / name)

    scripts = write_slurm_scripts_for_tree(tmp_path / "conv", frontier_cfg)

    assert len(scripts) == 3
    for s in scripts:
        assert s.name == "submit.sh"
        assert (s.parent / "pw.in").is_file()
        assert f"#SBATCH -J {s.parent.name}" in s.read_text()


def test_tree_empty_returns_empty_list(
    tmp_path: Path, frontier_cfg: SlurmConfig
) -> None:
    (tmp_path / "empty").mkdir()
    assert write_slurm_scripts_for_tree(tmp_path / "empty", frontier_cfg) == []
