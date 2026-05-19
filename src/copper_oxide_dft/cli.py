"""Command-line interface for the copper-oxide-dft pipeline.

Sub-commands are added as each phase lands. Currently:

* ``bulk-cu`` — generate a single bulk-Cu SCF input.
* ``sweep`` — Phase 1 convergence sweeps over ecutwfc / kpts / degauss.
* ``parse`` — read converged energies (etc.) from one or more pw.x outputs.
* ``inspect`` — decode a pw.in file and print a structural summary.
* ``make-slurm`` — emit submit.sh next to every pw.in under a directory.
* ``pourbaix`` — Phase 4 CHE Pourbaix for Cu / Cu2O / CuO.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from copper_oxide_dft import __version__
from copper_oxide_dft.analysis import (
    DEFAULT_CONVERGENCE_THRESHOLD_MEV_PER_ATOM,
    analyze_sweep,
    plot_convergence,
)
from copper_oxide_dft.che import (
    PhaseEnergetics,
    ReferenceEnergetics,
)
from copper_oxide_dft.convergence import (
    SUPPORTED_SWEEP_PARAMETERS,
    sweep_convergence,
)
from copper_oxide_dft.parse import parse_pw_output
from copper_oxide_dft.pourbaix import phase_diagram, plot_diagram
from copper_oxide_dft.qe_input import (
    DEFAULT_DEGAUSS_RY,
    DEFAULT_ECUTWFC_RY,
    DEFAULT_HUBBARD_U_CU_3D_EV,
    spin_and_hubbard_overrides,
    write_pw_input,
)
from copper_oxide_dft.structure_builder import (
    CU_LATTICE_PARAMETER_ANG,
    build_bulk_cu,
    build_bulk_cu2o,
    build_bulk_cuo,
    build_reference_h2,
    build_reference_h2o,
    summarize_layers,
)
from copper_oxide_dft.submit import SlurmConfig, write_slurm_scripts_for_tree

# Literature defaults for the Pourbaix CLI when the user has no DFT energies
# of their own yet. These are experimental Gibbs free energies of formation
# (NIST, 298 K) cast into the PhaseEnergetics/ReferenceEnergetics shape so
# that the CHE machinery reproduces the textbook Cu Pourbaix diagram. They
# get replaced by --energies <json> once Phase 1-2 calculations complete.
LITERATURE_REFERENCES_EV = {
    "e_h2_ev": 0.0,
    "e_h2o_ev": -2.458,
    "zpe_h2_ev": 0.0,
    "zpe_h2o_ev": 0.0,
    "ts_h2_ev": 0.0,
    "ts_h2o_ev": 0.0,
}
LITERATURE_PHASES_EV = [
    {"name": "Cu(metal)", "n_cu": 1, "n_o": 0, "e_dft_ev": 0.0},
    {"name": "Cu2O", "n_cu": 2, "n_o": 1, "e_dft_ev": -1.534},
    {"name": "CuO", "n_cu": 1, "n_o": 1, "e_dft_ev": -1.344},
]


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


@main.command("inspect")
@click.argument(
    "input_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--layer-tol",
    type=float,
    default=0.1,
    show_default=True,
    help="z-coordinate tolerance for grouping atoms into layers (A).",
)
def inspect_cmd(input_file: Path, layer_tol: float) -> None:
    """Decode a pw.x input file and print a structural summary.

    Use this before submitting jobs to verify cell, composition, and
    layer-by-layer atom positions. For slabs, the layer summary makes
    surface termination and depth ordering trivially visible.
    """
    import numpy as np
    from ase.io.espresso import read_espresso_in

    with input_file.open() as fh:
        atoms = read_espresso_in(fh)

    click.echo(f"File:        {input_file}")
    click.echo(f"Composition: {atoms.get_chemical_formula()} ({len(atoms)} atoms)")
    click.echo(f"Volume:      {atoms.get_volume():.4f} A^3")
    click.echo("Cell vectors (A):")
    for label, vec in zip("abc", atoms.cell, strict=True):
        norm = float(np.linalg.norm(vec))
        click.echo(
            f"  {label}: ({vec[0]:9.4f} {vec[1]:9.4f} {vec[2]:9.4f})  |{label}| = {norm:.4f}"
        )

    layers = summarize_layers(atoms, tol=layer_tol)
    click.echo(f"\nLayers grouped by z (tol={layer_tol} A):")
    for i, layer in enumerate(layers):
        click.echo(
            f"  [{i:2d}] z = {layer.z:8.4f} A  thickness = {layer.thickness:.4f} A"
            f"  {layer.composition_label()}  ({layer.total_atoms} atoms)"
        )


@main.command("make-slurm")
@click.argument(
    "root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option("--account", required=True, help="SLURM account (e.g. CHM999).")
@click.option(
    "--target",
    type=click.Choice(["frontier", "andes"]),
    default="frontier",
    show_default=True,
    help="Cluster preset. Frontier = AMD GPU; Andes = CPU.",
)
@click.option("--nodes", type=int, default=1, show_default=True)
@click.option("--walltime", default="1:00:00", show_default=True, help="HH:MM:SS.")
@click.option(
    "--qe-module",
    default=None,
    help=(
        "Override the QE module to load on the cluster (e.g. quantum-espresso/7.3-gpu)."
    ),
)
def make_slurm(
    root: Path,
    account: str,
    target: str,
    nodes: int,
    walltime: str,
    qe_module: str | None,
) -> None:
    """Emit submit.sh next to every pw.in under ROOT.

    Run on the cluster after copying the sweep tree over, then submit
    each script (e.g. `for d in */; do (cd "$d" && sbatch submit.sh); done`).
    """
    overrides: dict[str, object] = {"nodes": nodes, "walltime": walltime}
    if qe_module is not None:
        overrides["qe_module"] = qe_module
    if target == "frontier":
        cfg = SlurmConfig.for_frontier(account, **overrides)
    else:
        cfg = SlurmConfig.for_andes(account, **overrides)
    scripts = write_slurm_scripts_for_tree(root, cfg)
    if not scripts:
        click.echo(f"No pw.in files found under {root}", err=True)
        raise SystemExit(1)
    for s in scripts:
        click.echo(f"Wrote {s}")


@main.command("make-pourbaix-inputs")
@click.argument(
    "root",
    type=click.Path(file_okay=False, path_type=Path),
)
@click.option(
    "--pseudo-dir",
    type=click.Path(path_type=Path),
    help="Directory containing UPF files; defaults to $CUOXDFT_PSEUDO_DIR.",
)
@click.option(
    "--cu-pseudo",
    default="Cu.upf",
    show_default=True,
    help="UPF filename for Cu.",
)
@click.option(
    "--o-pseudo",
    default="O.upf",
    show_default=True,
    help="UPF filename for O.",
)
@click.option(
    "--h-pseudo",
    default="H.upf",
    show_default=True,
    help="UPF filename for H.",
)
@click.option(
    "--ecutwfc",
    type=float,
    default=DEFAULT_ECUTWFC_RY,
    show_default=True,
    help="Plane-wave cutoff (Ry); use the Phase 1 converged value.",
)
@click.option(
    "--hubbard-u",
    type=float,
    default=DEFAULT_HUBBARD_U_CU_3D_EV,
    show_default=True,
    help="Hubbard U on Cu 3d (eV) applied to oxide bulks.",
)
def make_pourbaix_inputs(
    root: Path,
    pseudo_dir: Path | None,
    cu_pseudo: str,
    o_pseudo: str,
    h_pseudo: str,
    ecutwfc: float,
    hubbard_u: float,
) -> None:
    """Generate the QE inputs that feed the Phase 4 Pourbaix analysis.

    Writes one pw.in per system under ROOT:

      ROOT/bulk_cu/pw.in            (vc-relax, non-magnetic, no U)
      ROOT/bulk_cu2o/pw.in          (vc-relax, non-magnetic, DFT+U on Cu 3d)
      ROOT/bulk_cuo/pw.in           (vc-relax, AFM spin-pol, DFT+U on Cu 3d)
      ROOT/mol_h2/pw.in             (scf, gamma point, non-magnetic)
      ROOT/mol_h2o/pw.in            (scf, gamma point, non-magnetic)

    Run ``make-slurm`` next to wrap each one in a submit.sh for Frontier.
    """
    root.mkdir(parents=True, exist_ok=True)

    # --- bulk Cu: vc-relax, no U, no spin ---
    cu = build_bulk_cu()
    write_pw_input(
        cu,
        out_path=root / "bulk_cu" / "pw.in",
        pseudopotentials={"Cu": cu_pseudo},
        pseudo_dir=pseudo_dir,
        calculation="vc-relax",
        prefix="bulk_cu",
        ecutwfc=ecutwfc,
        kpts=(8, 8, 8),
        degauss=DEFAULT_DEGAUSS_RY,
    )

    # --- bulk Cu2O: vc-relax, non-magnetic, DFT+U on Cu 3d ---
    cu2o = build_bulk_cu2o()
    cu2o_ov = spin_and_hubbard_overrides(
        cu2o, nspin=1, hubbard_u={"Cu": hubbard_u}
    )
    write_pw_input(
        cu2o,
        out_path=root / "bulk_cu2o" / "pw.in",
        pseudopotentials={"Cu": cu_pseudo, "O": o_pseudo},
        pseudo_dir=pseudo_dir,
        calculation="vc-relax",
        prefix="bulk_cu2o",
        ecutwfc=ecutwfc,
        kpts=(6, 6, 6),
        degauss=DEFAULT_DEGAUSS_RY,
        extra_input_data=cu2o_ov.namelist_overrides,
        additional_cards=cu2o_ov.hubbard_card,
    )

    # --- bulk CuO: vc-relax, AFM (nspin=2 with per-atom moments), DFT+U ---
    cuo = build_bulk_cuo()
    # Per-atom magnetizations are already on `cuo` from the builder; ASE
    # writes them as per-atom starting_magnetization cards. We still need
    # nspin=2 and the Hubbard U via the override helper.
    cuo_ov = spin_and_hubbard_overrides(
        cuo, nspin=2, hubbard_u={"Cu": hubbard_u}
    )
    write_pw_input(
        cuo,
        out_path=root / "bulk_cuo" / "pw.in",
        pseudopotentials={"Cu": cu_pseudo, "O": o_pseudo},
        pseudo_dir=pseudo_dir,
        calculation="vc-relax",
        prefix="bulk_cuo",
        ecutwfc=ecutwfc,
        kpts=(4, 6, 4),
        degauss=DEFAULT_DEGAUSS_RY,
        extra_input_data=cuo_ov.namelist_overrides,
        additional_cards=cuo_ov.hubbard_card,
    )

    # --- reference H2 molecule: gamma point, non-magnetic ---
    h2 = build_reference_h2()
    write_pw_input(
        h2,
        out_path=root / "mol_h2" / "pw.in",
        pseudopotentials={"H": h_pseudo},
        pseudo_dir=pseudo_dir,
        calculation="scf",
        prefix="mol_h2",
        ecutwfc=ecutwfc,
        kpts=(1, 1, 1),
        degauss=DEFAULT_DEGAUSS_RY,
    )

    # --- reference H2O molecule: gamma point, non-magnetic ---
    h2o = build_reference_h2o()
    write_pw_input(
        h2o,
        out_path=root / "mol_h2o" / "pw.in",
        pseudopotentials={"O": o_pseudo, "H": h_pseudo},
        pseudo_dir=pseudo_dir,
        calculation="scf",
        prefix="mol_h2o",
        ecutwfc=ecutwfc,
        kpts=(1, 1, 1),
        degauss=DEFAULT_DEGAUSS_RY,
    )

    click.echo(f"Wrote 5 systems under {root}:")
    for sub in ("bulk_cu", "bulk_cu2o", "bulk_cuo", "mol_h2", "mol_h2o"):
        click.echo(f"  {root / sub / 'pw.in'}")
    click.echo(
        f"Next: 'copper-oxide-dft make-slurm {root} --account=<your-account>' "
        "to write Frontier submit scripts."
    )


@main.command("sweep-analyze")
@click.argument(
    "root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--threshold-mev",
    type=float,
    default=DEFAULT_CONVERGENCE_THRESHOLD_MEV_PER_ATOM,
    show_default=True,
    help="Convergence threshold (meV/atom). Phase 1 success criterion is 1 meV/atom.",
)
@click.option(
    "--png",
    "png_path",
    type=click.Path(path_type=Path),
    help="Optional path to save the convergence plot.",
)
def sweep_analyze(root: Path, threshold_mev: float, png_path: Path | None) -> None:
    """Parse a convergence-sweep tree and pick the converged parameter value.

    Walks ROOT (produced by ``copper-oxide-dft sweep``), parses each
    ``pw.out``, prints a per-point energy table, reports the smallest
    parameter value that converges total energy per atom to within
    ``--threshold-mev``, and optionally renders a convergence plot.
    """
    result = analyze_sweep(root, threshold_mev_per_atom=threshold_mev)
    click.echo(
        f"Sweep parameter: {result.param_name}  "
        f"(threshold {result.threshold_mev_per_atom} meV/atom)"
    )
    asymptote_point = (
        result.points[0] if result.low_value_is_asymptote else result.points[-1]
    )
    asymptote_ev = asymptote_point.energy_per_atom_ev
    click.echo(
        f"{'value':>12}  {'E/atom (eV)':>16}  {'ΔE (meV/atom)':>16}  done"
    )
    for point in result.points:
        delta_mev = (point.energy_per_atom_ev - asymptote_ev) * 1.0e3
        click.echo(
            f"{point.param_value:>12g}  {point.energy_per_atom_ev:>16.6f}  "
            f"{delta_mev:>+16.3f}  {point.job_done}"
        )
    if result.converged_value is None:
        click.echo(
            "Converged value: NONE within threshold (extend the sweep upward "
            "or relax the threshold).",
            err=True,
        )
        raise SystemExit(1)
    click.echo(
        f"Converged value: {result.param_name} = {result.converged_value:g}"
    )

    if png_path is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ax = plot_convergence(result)
        ax.figure.tight_layout()
        ax.figure.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(ax.figure)
        click.echo(f"Wrote {png_path}")


@main.command("aggregate-pourbaix-energies")
@click.argument(
    "root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(path_type=Path),
    required=True,
    help="JSON output path (consumable by 'pourbaix --energies').",
)
@click.option(
    "--require-job-done/--allow-incomplete",
    default=True,
    show_default=True,
    help=(
        "By default refuse to aggregate if any pw.out is missing the 'JOB DONE.' "
        "marker. Pass --allow-incomplete to override (energies will be the last "
        "value reported, which may not be converged)."
    ),
)
def aggregate_pourbaix_energies(
    root: Path, out_path: Path, require_job_done: bool
) -> None:
    """Aggregate pw.out files under ROOT into a JSON for the pourbaix CLI.

    Expects the directory layout produced by ``make-pourbaix-inputs``:
    ROOT/bulk_cu/pw.out, ROOT/bulk_cu2o/pw.out, ROOT/bulk_cuo/pw.out,
    ROOT/mol_h2/pw.out, ROOT/mol_h2o/pw.out.

    The output JSON has the schema consumed by ``pourbaix --energies``:
    a ``references`` block with H2/H2O total energies and a ``phases``
    block with per-formula-unit DFT energies for each solid phase.

    Run on the cluster (or wherever pw.out files live) once the five
    Frontier jobs finish; then pass the result to ``pourbaix --energies``.
    """
    payload = _aggregate_pourbaix_energies(root, require_job_done=require_job_done)
    out_path.write_text(json.dumps(payload, indent=2))
    click.echo(f"Wrote {out_path}")
    refs = payload["references"]
    click.echo(
        f"  references: E(H2)={refs['e_h2_ev']:+.4f} eV  "
        f"E(H2O)={refs['e_h2o_ev']:+.4f} eV"
    )
    for phase in payload["phases"]:
        click.echo(
            f"  {phase['name']:>10}: E_DFT={phase['e_dft_ev']:+.4f} eV  "
            f"(n_cu={phase['n_cu']}, n_o={phase['n_o']})"
        )


def _aggregate_pourbaix_energies(
    root: Path, *, require_job_done: bool
) -> dict[str, Any]:
    """Walk a make-pourbaix-inputs tree and build the --energies JSON.

    Pulled out of the click command so it stays unit-testable.
    """
    # The five expected systems and their per-formula-unit stoichiometry.
    # We parse total energies as written by QE and divide by the count of
    # formula units in the conventional cell that build_bulk_* produces:
    #   bulk_cu (1 atom = 1 formula unit), bulk_cu2o (6 atoms = 2 f.u.),
    #   bulk_cuo (8 atoms = 4 f.u.). Molecules are one f.u. each.
    phase_specs = [
        # (subdir, json name, n_cu per f.u., n_o per f.u., atoms per f.u.,
        #  atoms in conventional cell from build_bulk_*)
        ("bulk_cu", "Cu(metal)", 1, 0, 1, 1),
        ("bulk_cu2o", "Cu2O", 2, 1, 3, 6),
        ("bulk_cuo", "CuO", 1, 1, 2, 8),
    ]

    def _parse_one(subdir: str) -> float:
        out_file = root / subdir / "pw.out"
        if not out_file.is_file():
            raise click.ClickException(f"Missing pw.out: {out_file}")
        result = parse_pw_output(out_file)
        if require_job_done and not result.job_done:
            raise click.ClickException(
                f"{out_file} did not finish (no 'JOB DONE.' marker). "
                "Pass --allow-incomplete to skip this check."
            )
        return float(result.total_energy_ev)

    phases_payload: list[dict[str, Any]] = []
    for subdir, name, n_cu, n_o, atoms_per_fu, atoms_in_cell in phase_specs:
        e_cell_ev = _parse_one(subdir)
        n_formula_units = atoms_in_cell // atoms_per_fu
        e_per_fu_ev = e_cell_ev / n_formula_units
        phases_payload.append(
            {
                "name": name,
                "n_cu": n_cu,
                "n_o": n_o,
                "e_dft_ev": e_per_fu_ev,
            }
        )

    return {
        "references": {
            "e_h2_ev": _parse_one("mol_h2"),
            "e_h2o_ev": _parse_one("mol_h2o"),
        },
        "phases": phases_payload,
    }


@main.command("pourbaix")
@click.option(
    "--u",
    "u_marker_v",
    type=float,
    help="Mark this potential (V vs. SHE) on the diagram and report stable phase.",
)
@click.option(
    "--ph",
    "ph_marker",
    type=float,
    help="Mark this pH on the diagram. Must be paired with --u.",
)
@click.option(
    "--u-range",
    "u_range_str",
    default="-1.0,1.0",
    show_default=True,
    help="Comma-separated min,max potential (V vs. SHE).",
)
@click.option(
    "--ph-range",
    "ph_range_str",
    default="0,14",
    show_default=True,
    help="Comma-separated min,max pH.",
)
@click.option(
    "--u-steps",
    type=int,
    default=81,
    show_default=True,
    help="Number of U grid points (inclusive endpoints).",
)
@click.option(
    "--ph-steps",
    type=int,
    default=71,
    show_default=True,
    help="Number of pH grid points (inclusive endpoints).",
)
@click.option(
    "--energies",
    "energies_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help=(
        "JSON file with our own DFT phase + reference energies. If omitted, "
        "experimental ΔG_f values (NIST, 298 K) are used as placeholders so "
        "the diagram is qualitatively correct but not based on this project's "
        "calculations. Schema: {references: {e_h2_ev, e_h2o_ev, [zpe/ts]}, "
        "phases: [{name, n_cu, n_o, e_dft_ev, [zpe_ev, ts_ev]}, ...]}."
    ),
)
@click.option(
    "--png",
    "png_path",
    type=click.Path(path_type=Path),
    help="Save the Pourbaix plot to this PNG file.",
)
@click.option(
    "--json",
    "json_path",
    type=click.Path(path_type=Path),
    help="Save the grid (U, pH, stable-phase index, per-phase ΔG) to this JSON file.",
)
def pourbaix_cmd(
    u_marker_v: float | None,
    ph_marker: float | None,
    u_range_str: str,
    ph_range_str: str,
    u_steps: int,
    ph_steps: int,
    energies_path: Path | None,
    png_path: Path | None,
    json_path: Path | None,
) -> None:
    """Build a Cu / Cu2O / CuO Pourbaix diagram via the Computational Hydrogen Electrode.

    Without --energies, literature ΔG_f values are used so the diagram is
    qualitatively correct out of the box. Once Phase 1-2 calculations
    complete, pass --energies <json> to substitute project DFT+U energies.

    Example: report the stable phase at -0.4 V vs. SHE, pH 7:

      copper-oxide-dft pourbaix --u -0.4 --ph 7 --png pourbaix.png
    """
    if (u_marker_v is None) != (ph_marker is None):
        raise click.UsageError("--u and --ph must be given together.")

    u_range = _parse_pair(u_range_str, "u-range")
    ph_range = _parse_pair(ph_range_str, "ph-range")

    references, phases = _load_pourbaix_inputs(energies_path)

    # The Cu-metal reference is the phase with n_o = 0; it must be present
    # in the phase list for ΔG_per_Cu = 0 to appear as a region.
    cu_metal_candidates = [p for p in phases if p.n_o == 0]
    if not cu_metal_candidates:
        raise click.UsageError(
            "Phase list must include a Cu-metal entry (n_o=0) to serve as the "
            "per-Cu free-energy zero."
        )
    cu_metal_reference = cu_metal_candidates[0]

    diagram = phase_diagram(
        phases,
        cu_metal_reference,
        references,
        u_range_v=u_range,
        u_steps=u_steps,
        ph_range=ph_range,
        ph_steps=ph_steps,
    )

    if u_marker_v is not None and ph_marker is not None:
        stable = diagram.stable_phase_at(u_marker_v, ph_marker)
        click.echo(
            f"At U = {u_marker_v:+.3f} V vs. SHE, pH = {ph_marker}: "
            f"stable phase = {stable}"
        )
        # Also report per-phase ΔG/Cu at the marked point so a near-tie is visible.
        import numpy as np

        i_u = int(np.argmin(np.abs(diagram.u_grid_v - u_marker_v)))
        i_ph = int(np.argmin(np.abs(diagram.ph_grid - ph_marker)))
        for i, name in enumerate(diagram.phase_names):
            click.echo(
                f"  ΔG_per_Cu({name}) = "
                f"{diagram.free_energies_per_cu_ev[i, i_ph, i_u]:+.3f} eV"
            )

    if png_path is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        mark = (
            (u_marker_v, ph_marker)
            if u_marker_v is not None and ph_marker is not None
            else None
        )
        ax = plot_diagram(diagram, mark_point=mark)
        ax.figure.tight_layout()
        ax.figure.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(ax.figure)
        click.echo(f"Wrote {png_path}")

    if json_path is not None:
        json_path.write_text(
            json.dumps(
                {
                    "u_grid_v": diagram.u_grid_v.tolist(),
                    "ph_grid": diagram.ph_grid.tolist(),
                    "phase_names": list(diagram.phase_names),
                    "free_energies_per_cu_ev": diagram.free_energies_per_cu_ev.tolist(),
                    "stable_phase_index": diagram.stable_phase_index.tolist(),
                },
                indent=2,
            )
        )
        click.echo(f"Wrote {json_path}")


def _parse_pair(text: str, label: str) -> tuple[float, float]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        raise click.UsageError(f"--{label} must be 'min,max'; got {text!r}.")
    try:
        return (float(parts[0]), float(parts[1]))
    except ValueError as exc:
        raise click.UsageError(f"--{label} must be numeric; got {text!r}.") from exc


def _load_pourbaix_inputs(
    energies_path: Path | None,
) -> tuple[ReferenceEnergetics, list[PhaseEnergetics]]:
    if energies_path is None:
        ref_dict = LITERATURE_REFERENCES_EV
        phases_data = LITERATURE_PHASES_EV
    else:
        loaded = json.loads(energies_path.read_text())
        ref_dict = loaded["references"]
        phases_data = loaded["phases"]

    references = ReferenceEnergetics(**ref_dict)
    phases = [PhaseEnergetics(**p) for p in phases_data]
    return references, phases


if __name__ == "__main__":
    main()
