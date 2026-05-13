"""Command-line interface for the copper-oxide-dft pipeline.

Sub-commands are added as each phase lands. Currently:

* ``bulk-cu`` — generate a single bulk-Cu SCF input.
* ``sweep`` — Phase 1 convergence sweeps over ecutwfc / kpts / degauss.
* ``parse`` — read converged energies (etc.) from one or more pw.x outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from copper_oxide_dft import __version__
from copper_oxide_dft.convergence import (
    SUPPORTED_SWEEP_PARAMETERS,
    sweep_convergence,
)
from copper_oxide_dft.parse import parse_pw_output
from copper_oxide_dft.qe_input import (
    DEFAULT_DEGAUSS_RY,
    DEFAULT_ECUTWFC_RY,
    write_pw_input,
)
from copper_oxide_dft.structure_builder import CU_LATTICE_PARAMETER_ANG, build_bulk_cu


@click.group()
@click.version_option(version=__version__)
def main() -> None:
    """copper-oxide-dft: DFT workflow for Cu oxide on Cu surfaces."""


@main.command("bulk-cu")
@click.option(
    "--out",
    "out_path",
    type=click.Path(path_type=Path),
    required=True,
    help="Output path for the pw.x input file.",
)
@click.option(
    "--pseudo",
    "pseudo_filename",
    default="Cu.upf",
    show_default=True,
    help="UPF filename for Cu (must exist in --pseudo-dir).",
)
@click.option(
    "--pseudo-dir",
    type=click.Path(path_type=Path),
    help="Directory containing UPF files; defaults to $CUOXDFT_PSEUDO_DIR.",
)
@click.option(
    "--calculation",
    type=click.Choice(["scf", "relax", "vc-relax"]),
    default="scf",
    show_default=True,
    help="pw.x calculation type. Use vc-relax for lattice-parameter optimization.",
)
@click.option(
    "--a",
    "lattice_a",
    type=float,
    default=CU_LATTICE_PARAMETER_ANG,
    show_default=True,
    help="fcc lattice parameter (A).",
)
@click.option(
    "--ecutwfc",
    type=float,
    default=DEFAULT_ECUTWFC_RY,
    show_default=True,
    help="Plane-wave cutoff (Ry).",
)
@click.option(
    "--kpts",
    "kpts_n",
    type=int,
    default=8,
    show_default=True,
    help="Monkhorst-Pack grid size (used for all three directions).",
)
@click.option(
    "--degauss",
    type=float,
    default=DEFAULT_DEGAUSS_RY,
    show_default=True,
    help="Marzari-Vanderbilt smearing width (Ry).",
)
def bulk_cu(
    out_path: Path,
    pseudo_filename: str,
    pseudo_dir: Path | None,
    calculation: str,
    lattice_a: float,
    ecutwfc: float,
    kpts_n: int,
    degauss: float,
) -> None:
    """Generate a pw.x input file for bulk fcc Cu."""
    atoms = build_bulk_cu(a=lattice_a)
    written = write_pw_input(
        atoms,
        out_path=out_path,
        pseudopotentials={"Cu": pseudo_filename},
        calculation=calculation,
        prefix="bulk_cu",
        ecutwfc=ecutwfc,
        kpts=(kpts_n, kpts_n, kpts_n),
        degauss=degauss,
        pseudo_dir=pseudo_dir,
    )
    click.echo(f"Wrote {written}")


@main.command("sweep")
@click.option(
    "--param",
    type=click.Choice(sorted(SUPPORTED_SWEEP_PARAMETERS)),
    required=True,
    help="Parameter to sweep.",
)
@click.option(
    "--values",
    required=True,
    help="Comma-separated sweep values (e.g. 40,60,80,100).",
)
@click.option(
    "--out",
    "out_root",
    type=click.Path(path_type=Path),
    required=True,
    help="Root directory for the generated input tree.",
)
@click.option(
    "--pseudo",
    "pseudo_filename",
    default="Cu.upf",
    show_default=True,
)
@click.option(
    "--pseudo-dir",
    type=click.Path(path_type=Path),
    help="Directory containing UPF files; defaults to $CUOXDFT_PSEUDO_DIR.",
)
@click.option(
    "--a",
    "lattice_a",
    type=float,
    default=CU_LATTICE_PARAMETER_ANG,
    show_default=True,
    help="fcc lattice parameter (A) for the bulk Cu structure.",
)
def sweep(
    param: str,
    values: str,
    out_root: Path,
    pseudo_filename: str,
    pseudo_dir: Path | None,
    lattice_a: float,
) -> None:
    """Generate a convergence-sweep tree of pw.x SCF inputs for bulk Cu."""
    parsed_values: list[float] = [
        float(v.strip()) for v in values.split(",") if v.strip()
    ]
    atoms = build_bulk_cu(a=lattice_a)
    written = sweep_convergence(
        atoms,
        out_root=out_root,
        pseudopotentials={"Cu": pseudo_filename},
        param=param,
        values=parsed_values,
        pseudo_dir=pseudo_dir,
    )
    for path in written:
        click.echo(f"Wrote {path}")


@main.command("parse")
@click.argument(
    "outputs",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit machine-readable JSON instead of a table.",
)
def parse_cmd(outputs: tuple[Path, ...], as_json: bool) -> None:
    """Parse one or more pw.x stdout files and print scalar results."""
    rows: list[dict[str, object]] = []
    for path in outputs:
        result = parse_pw_output(path)
        rows.append(
            {
                "path": str(path),
                "total_energy_ry": result.total_energy_ry,
                "total_energy_ev": result.total_energy_ev,
                "fermi_energy_ev": result.fermi_energy_ev,
                "total_magnetization_bohr": result.total_magnetization_bohr,
                "job_done": result.job_done,
            }
        )
    if as_json:
        click.echo(json.dumps(rows, indent=2))
        return
    for row in rows:
        click.echo(
            f"{row['path']}  E={row['total_energy_ry']:.6f} Ry  "
            f"({row['total_energy_ev']:.4f} eV)  "
            f"E_F={row['fermi_energy_ev']}  mag={row['total_magnetization_bohr']}  "
            f"done={row['job_done']}"
        )


if __name__ == "__main__":
    main()
