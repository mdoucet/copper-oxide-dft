"""SLURM submission scaffolding for ORNL clusters (Frontier and Andes).

Generates a ``submit.sh`` next to each ``pw.in`` so the user can ``sbatch``
it on the cluster. Defaults are GPU-flavored for ORNL Frontier (8 GCDs
per node, GPU-aware MPI, ``srun`` with closest-binding); an Andes (CPU)
factory is provided for the development cluster.

The submitter does **not** invoke ``sbatch`` itself — Python on the login
node would be unusual, and keeping this side-effect-free makes the
behavior testable. Users run ``sbatch submit.sh`` (or
``for d in */; do (cd "$d" && sbatch submit.sh); done`` for a sweep).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SlurmConfig:
    """Cluster-side SLURM parameters for a single ``pw.x`` run.

    The class supports both CPU and GPU clusters. GPU fields
    (``gpus_per_node``, ``cpus_per_task``, ``gpu_bind``,
    ``gpu_aware_mpi``) are optional; when unset, no GPU-related SBATCH
    or environment lines are emitted. Use :meth:`for_frontier` or
    :meth:`for_andes` for the standard ORNL cluster presets.
    """

    account: str
    partition: str = "batch"
    nodes: int = 1
    ntasks_per_node: int = 32
    walltime: str = "1:00:00"
    qe_module: str = "quantum-espresso"
    pw_executable: str = "pw.x"
    mpi_launcher: str = "srun"
    extra_modules: Sequence[str] = field(default_factory=tuple)
    extra_exports: Mapping[str, str] = field(default_factory=dict)
    # GPU-related (Frontier and similar); leave None on CPU clusters.
    gpus_per_node: int | None = None
    cpus_per_task: int | None = None
    gpu_bind: str | None = None
    gpu_aware_mpi: bool = False

    @classmethod
    def for_frontier(cls, account: str, **overrides: Any) -> SlurmConfig:
        """Preset for ORNL Frontier (4x MI250X = 8 GCDs/node, AMD EPYC 64-core).

        Defaults emit one MPI rank per GCD with 7 cores each, GPU-aware
        MPICH, and ``--gpu-bind=closest``. Override any field via kwargs
        (e.g. ``walltime="2:00:00"``).
        """
        base = cls(
            account=account,
            partition="batch",
            ntasks_per_node=8,
            cpus_per_task=7,
            gpus_per_node=8,
            gpu_bind="closest",
            gpu_aware_mpi=True,
            extra_modules=("PrgEnv-gnu", "rocm"),
        )
        return replace(base, **overrides) if overrides else base

    @classmethod
    def for_andes(cls, account: str, **overrides: Any) -> SlurmConfig:
        """Preset for ORNL Andes (CPU, 32 cores/node)."""
        base = cls(account=account, partition="batch", ntasks_per_node=32)
        return replace(base, **overrides) if overrides else base


def _render_sbatch_lines(cfg: SlurmConfig, job_name: str) -> str:
    lines = [
        f"#SBATCH -A {cfg.account}",
        f"#SBATCH -J {job_name}",
        f"#SBATCH -p {cfg.partition}",
        f"#SBATCH -N {cfg.nodes}",
        f"#SBATCH --ntasks-per-node={cfg.ntasks_per_node}",
        f"#SBATCH -t {cfg.walltime}",
    ]
    if cfg.cpus_per_task is not None:
        lines.append(f"#SBATCH -c {cfg.cpus_per_task}")
    if cfg.gpus_per_node is not None:
        lines.append(f"#SBATCH --gpus-per-node={cfg.gpus_per_node}")
    lines.extend(["#SBATCH -o slurm-%j.out", "#SBATCH -e slurm-%j.err"])
    return "\n".join(lines)


def _render_exports(cfg: SlurmConfig) -> str:
    exports = dict(cfg.extra_exports)
    if cfg.cpus_per_task is not None:
        exports.setdefault("OMP_NUM_THREADS", "$SLURM_CPUS_PER_TASK")
    if cfg.gpu_aware_mpi:
        exports.setdefault("MPICH_GPU_SUPPORT_ENABLED", "1")
    if not exports:
        return ""
    return "\n".join(f'export {k}="{v}"' for k, v in exports.items()) + "\n\n"


def _render_srun_args(cfg: SlurmConfig) -> str:
    parts: list[str] = []
    if cfg.gpus_per_node is not None:
        parts.append("--gpus-per-task=1")
    if cfg.gpu_bind:
        parts.append(f"--gpu-bind={cfg.gpu_bind}")
    return (" " + " ".join(parts)) if parts else ""


def _render_script(
    cfg: SlurmConfig,
    *,
    job_name: str,
    input_file_name: str,
    output_file_name: str,
) -> str:
    # Cray convention: PrgEnv/runtime modules load before the application.
    module_lines = "\n".join(
        f"module load {m}" for m in (*cfg.extra_modules, cfg.qe_module)
    )
    return f"""\
#!/bin/bash
{_render_sbatch_lines(cfg, job_name)}

set -euo pipefail

module purge
{module_lines}

{_render_exports(cfg)}cd "$SLURM_SUBMIT_DIR"
{cfg.mpi_launcher}{_render_srun_args(cfg)} {cfg.pw_executable} \
-in {input_file_name} > {output_file_name}
"""


def write_slurm_script(
    work_dir: str | Path,
    cfg: SlurmConfig,
    *,
    job_name: str | None = None,
    input_file_name: str = "pw.in",
    output_file_name: str = "pw.out",
    script_name: str = "submit.sh",
) -> Path:
    """Write a single SLURM submission script next to a ``pw.in``.

    Args:
        work_dir: Directory containing the ``pw.in``. The script and
            ``pw.out`` will live in the same directory at runtime.
        cfg: Cluster parameters.
        job_name: ``-J`` value. Defaults to the parent directory name,
            which for sweeps gives readable names like ``ecutwfc_60``.
        input_file_name: Filename of the pw.x input inside ``work_dir``.
        output_file_name: Filename for the pw.x stdout capture.
        script_name: Filename for the generated submission script.

    Returns:
        Path to the written script.

    Raises:
        FileNotFoundError: If ``work_dir/input_file_name`` does not exist.
    """
    work = Path(work_dir).resolve()
    if not (work / input_file_name).is_file():
        raise FileNotFoundError(f"Expected {input_file_name} in {work}")

    job = job_name or work.name
    text = _render_script(
        cfg,
        job_name=job,
        input_file_name=input_file_name,
        output_file_name=output_file_name,
    )
    script_path = work / script_name
    script_path.write_text(text)
    script_path.chmod(0o755)
    return script_path


def write_slurm_scripts_for_tree(
    root: str | Path,
    cfg: SlurmConfig,
    *,
    input_file_name: str = "pw.in",
    output_file_name: str = "pw.out",
    script_name: str = "submit.sh",
) -> list[Path]:
    """Find every ``pw.in`` under ``root`` and emit a submit script beside each.

    Used after :func:`copper_oxide_dft.convergence.sweep_convergence` to
    turn a tree of inputs into a tree of submittable jobs.

    Args:
        root: Directory tree to scan recursively.
        cfg: Shared cluster parameters.

    Returns:
        Paths to the written scripts, one per discovered input.
    """
    return [
        write_slurm_script(
            in_path.parent,
            cfg,
            input_file_name=input_file_name,
            output_file_name=output_file_name,
            script_name=script_name,
        )
        for in_path in _find_inputs(Path(root), input_file_name)
    ]


def _find_inputs(root: Path, input_file_name: str) -> Iterator[Path]:
    yield from sorted(root.rglob(input_file_name))


__all__: Iterable[str] = (
    "SlurmConfig",
    "write_slurm_script",
    "write_slurm_scripts_for_tree",
)
