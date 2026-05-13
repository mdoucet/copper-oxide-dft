"""Command-line interface for the copper-oxide-dft pipeline.

Phase 0 only exposes ``bulk-cu`` (generate a bulk-Cu SCF input file). More
sub-commands appear as later phases land.
"""

from __future__ import annotations

from pathlib import Path

import click

from copper_oxide_dft import __version__
from copper_oxide_dft.qe_input import (
    DEFAULT_DEGAUSS_RY,
    DEFAULT_ECUTWFC_RY,
    write_scf_input,
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
    "--a",
    "lattice_a",
    type=float,
    default=CU_LATTICE_PARAMETER_ANG,
    show_default=True,
    help="fcc lattice parameter (Å).",
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
    lattice_a: float,
    ecutwfc: float,
    kpts_n: int,
    degauss: float,
) -> None:
    """Generate a pw.x SCF input file for bulk fcc Cu."""
    atoms = build_bulk_cu(a=lattice_a)
    written = write_scf_input(
        atoms,
        out_path=out_path,
        pseudopotentials={"Cu": pseudo_filename},
        prefix="bulk_cu",
        ecutwfc=ecutwfc,
        kpts=(kpts_n, kpts_n, kpts_n),
        degauss=degauss,
        pseudo_dir=pseudo_dir,
    )
    click.echo(f"Wrote {written}")


if __name__ == "__main__":
    main()
