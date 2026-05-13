"""SLURM submission scaffolding for ORNL Andes (and similar CPU clusters).

Generates a ``submit.sh`` next to each ``pw.in`` so the user can ``sbatch``
it on the cluster. Defaults target ORNL Andes (CPU, ``srun`` launcher,
batch partition). Frontier (GPU) will need a different template once we
get there; rather than over-generalize now, we'll add a sibling helper
when Phase 1 results are in hand.

The submitter does **not** invoke ``sbatch`` itself — Python on the login
node would be unusual, and keeping this side-effect-free makes the
behavior testable. Users run ``sbatch submit.sh`` (or
``for d in */; do (cd "$d" && sbatch submit.sh); done`` for a sweep).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SlurmConfig:
    """Cluster-side SLURM parameters for a single pw.x run."""

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


def _render_script(
    cfg: SlurmConfig,
    *,
    job_name: str,
    input_file_name: str,
    output_file_name: str,
) -> str:
    module_lines = "\n".join(
        f"module load {m}" for m in (cfg.qe_module, *cfg.extra_modules)
    )
    export_lines = "\n".join(f'export {k}="{v}"' for k, v in cfg.extra_exports.items())
    if export_lines:
        export_lines += "\n"

    return f"""\
#!/bin/bash
#SBATCH -A {cfg.account}
#SBATCH -J {job_name}
#SBATCH -p {cfg.partition}
#SBATCH -N {cfg.nodes}
#SBATCH --ntasks-per-node={cfg.ntasks_per_node}
#SBATCH -t {cfg.walltime}
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -euo pipefail

module purge
{module_lines}

{export_lines}cd "$SLURM_SUBMIT_DIR"
{cfg.mpi_launcher} {cfg.pw_executable} -in {input_file_name} > {output_file_name}
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
