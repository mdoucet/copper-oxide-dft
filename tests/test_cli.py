"""Tests for the copper-oxide-dft CLI."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from copper_oxide_dft import __version__
from copper_oxide_dft.cli import main


def test_cli_version() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_cli_bulk_cu_writes_input_file(tmp_path: Path) -> None:
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()
    (pseudo_dir / "Cu.upf").write_text("")  # placeholder UPF; QE itself is not invoked

    out_file = tmp_path / "bulk_cu.in"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "bulk-cu",
            "--out",
            str(out_file),
            "--pseudo-dir",
            str(pseudo_dir),
            "--pseudo",
            "Cu.upf",
            "--ecutwfc",
            "60",
            "--kpts",
            "6",
        ],
    )
    assert result.exit_code == 0, result.output
    assert out_file.is_file()
    contents = out_file.read_text()
    assert "ecutwfc" in contents.lower()
    assert "K_POINTS" in contents


def test_cli_sweep_creates_tree_of_inputs(tmp_path: Path) -> None:
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()
    (pseudo_dir / "Cu.upf").write_text("")

    out_root = tmp_path / "conv"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "sweep",
            "--param",
            "ecutwfc",
            "--values",
            "40,60,80",
            "--out",
            str(out_root),
            "--pseudo-dir",
            str(pseudo_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    for value in (40, 60, 80):
        assert (out_root / f"ecutwfc_{value}" / "pw.in").is_file()


def test_cli_inspect_summarizes_bulk_cu(tmp_path: Path) -> None:
    """End-to-end: generate a bulk-Cu pw.in via the CLI, then inspect it."""
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()
    (pseudo_dir / "Cu.upf").write_text("")

    pw_in = tmp_path / "bulk_cu.in"
    runner = CliRunner()
    assert (
        runner.invoke(
            main,
            [
                "bulk-cu",
                "--out",
                str(pw_in),
                "--pseudo-dir",
                str(pseudo_dir),
            ],
        ).exit_code
        == 0
    )

    result = runner.invoke(main, ["inspect", str(pw_in)])
    assert result.exit_code == 0, result.output
    assert "Composition: Cu (1 atoms)" in result.output
    assert "Cell vectors" in result.output
    assert "Cux1" in result.output  # one Cu atom in one layer


def test_cli_make_slurm_defaults_to_frontier(tmp_path: Path) -> None:
    for name in ("ecutwfc_40", "ecutwfc_60"):
        d = tmp_path / "conv" / name
        d.mkdir(parents=True)
        (d / "pw.in").write_text("&CONTROL\n/\n")

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "make-slurm",
            str(tmp_path / "conv"),
            "--account",
            "CHM999",
            "--walltime",
            "0:30:00",
        ],
    )
    assert result.exit_code == 0, result.output
    sh = tmp_path / "conv" / "ecutwfc_40" / "submit.sh"
    text = sh.read_text()
    assert "#SBATCH -A CHM999" in text
    assert "#SBATCH -t 0:30:00" in text
    # Frontier defaults
    assert "#SBATCH --gpus-per-node=8" in text
    assert "--gpu-bind=closest" in text


def test_cli_make_slurm_andes_target_omits_gpu_lines(tmp_path: Path) -> None:
    d = tmp_path / "run"
    d.mkdir()
    (d / "pw.in").write_text("&CONTROL\n/\n")
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "make-slurm",
            str(tmp_path),
            "--account",
            "CHM999",
            "--target",
            "andes",
        ],
    )
    assert result.exit_code == 0, result.output
    text = (d / "submit.sh").read_text()
    assert "--gpus-per-node" not in text
    assert "#SBATCH --ntasks-per-node=32" in text


def test_cli_make_slurm_qe_module_override(tmp_path: Path) -> None:
    d = tmp_path / "run"
    d.mkdir()
    (d / "pw.in").write_text("&CONTROL\n/\n")
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "make-slurm",
            str(tmp_path),
            "--account",
            "CHM999",
            "--qe-module",
            "quantum-espresso/7.3-gpu",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "module load quantum-espresso/7.3-gpu" in (d / "submit.sh").read_text()


def test_cli_make_slurm_errors_when_no_inputs(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["make-slurm", str(tmp_path / "empty"), "--account", "CHM999"],
    )
    assert result.exit_code != 0
    assert "No pw.in" in result.output


def test_cli_parse_emits_json(tmp_path: Path) -> None:
    output = tmp_path / "scf.out"
    output.write_text(
        "!    total energy = -100.0 Ry\nthe Fermi energy is 3.0 ev\nJOB DONE.\n"
    )
    runner = CliRunner()
    result = runner.invoke(main, ["parse", "--json", str(output)])
    assert result.exit_code == 0, result.output
    assert '"total_energy_ry": -100.0' in result.output
    assert '"job_done": true' in result.output


def test_cli_pourbaix_reports_stable_phase_at_minus_0p4_v_ph7() -> None:
    """Smoke test for the headline use case: CHE Pourbaix with literature defaults."""
    runner = CliRunner()
    result = runner.invoke(main, ["pourbaix", "--u", "-0.4", "--ph", "7"])
    assert result.exit_code == 0, result.output
    assert "stable phase = Cu(metal)" in result.output
    # Per-phase ΔG breakdown should also be visible.
    assert "ΔG_per_Cu(Cu(metal))" in result.output
    assert "ΔG_per_Cu(Cu2O)" in result.output
    assert "ΔG_per_Cu(CuO)" in result.output


def test_cli_pourbaix_saves_png_and_json(tmp_path: Path) -> None:
    png = tmp_path / "diagram.png"
    json_path = tmp_path / "diagram.json"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "pourbaix",
            "--u",
            "-0.4",
            "--ph",
            "7",
            "--png",
            str(png),
            "--json",
            str(json_path),
            "--u-steps",
            "21",
            "--ph-steps",
            "15",
        ],
    )
    assert result.exit_code == 0, result.output
    assert png.is_file()
    assert json_path.is_file()
    import json as _json

    payload = _json.loads(json_path.read_text())
    assert payload["phase_names"] == ["Cu(metal)", "Cu2O", "CuO"]
    assert len(payload["u_grid_v"]) == 21
    assert len(payload["ph_grid"]) == 15


def test_cli_pourbaix_accepts_custom_energies_json(tmp_path: Path) -> None:
    """User supplies their own DFT+U energies once Frontier jobs finish."""
    import json as _json

    energies = tmp_path / "energies.json"
    energies.write_text(
        _json.dumps(
            {
                "references": {"e_h2_ev": 0.0, "e_h2o_ev": -2.458},
                "phases": [
                    {"name": "Cu", "n_cu": 1, "n_o": 0, "e_dft_ev": 0.0},
                    {"name": "Cu2O", "n_cu": 2, "n_o": 1, "e_dft_ev": -1.534},
                    {"name": "CuO", "n_cu": 1, "n_o": 1, "e_dft_ev": -1.344},
                ],
            }
        )
    )
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["pourbaix", "--u", "-0.4", "--ph", "7", "--energies", str(energies)],
    )
    assert result.exit_code == 0, result.output
    assert "stable phase = Cu" in result.output


def test_cli_pourbaix_errors_on_unpaired_u_or_ph() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["pourbaix", "--u", "-0.4"])
    assert result.exit_code != 0
    assert "must be given together" in result.output


def test_cli_make_pourbaix_inputs_writes_full_tree(tmp_path: Path) -> None:
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()
    for f in ("Cu.upf", "O.upf", "H.upf"):
        (pseudo_dir / f).write_text("")

    root = tmp_path / "phase4"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "make-pourbaix-inputs",
            str(root),
            "--pseudo-dir",
            str(pseudo_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    for sub in ("bulk_cu", "bulk_cu2o", "bulk_cuo", "mol_h2", "mol_h2o"):
        assert (root / sub / "pw.in").is_file(), f"missing {sub}/pw.in"


def test_cli_make_pourbaix_inputs_cuo_input_has_spin_and_hubbard(
    tmp_path: Path,
) -> None:
    """CuO is the magnetic + DFT+U case; both must end up in the namelist."""
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()
    for f in ("Cu.upf", "O.upf", "H.upf"):
        (pseudo_dir / f).write_text("")

    root = tmp_path / "phase4"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "make-pourbaix-inputs",
            str(root),
            "--pseudo-dir",
            str(pseudo_dir),
            "--hubbard-u",
            "4.5",
        ],
    )
    assert result.exit_code == 0, result.output
    cuo_text = (root / "bulk_cuo" / "pw.in").read_text()
    cuo_lower = cuo_text.lower()
    assert "nspin" in cuo_lower
    # AFM splitting → both Cu sub-species (Cu, Cu1) must carry the U term
    # in the QE 7.1+ HUBBARD card (the old Hubbard_U(i) namelist syntax is
    # gone — emitting it would make QE 7.1+ abort with "DFT+Hubbard
    # input syntax has changed since v7.1").
    assert "U Cu-3d 4.500000" in cuo_text
    assert "U Cu1-3d 4.500000" in cuo_text
    assert "hubbard_u(1)" not in cuo_lower
    # Per-atom starting magnetizations come through ASE for the AFM ordering.
    assert "starting_magnetization" in cuo_lower
    # And the default projector is ortho-atomic (the 2026-05-19 calibration).
    assert "HUBBARD { ortho-atomic }" in cuo_text


def test_cli_make_pourbaix_inputs_projector_type_flag(tmp_path: Path) -> None:
    """--projector-type threads through to the HUBBARD card.

    Uses ``atomic`` because it is the now-non-default value — verifies
    the flag does something, not that the default does what it should
    (the test above already covers the default path).
    """
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()
    for f in ("Cu.upf", "O.upf", "H.upf"):
        (pseudo_dir / f).write_text("")

    root = tmp_path / "phase4"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "make-pourbaix-inputs",
            str(root),
            "--pseudo-dir",
            str(pseudo_dir),
            "--projector-type",
            "atomic",
        ],
    )
    assert result.exit_code == 0, result.output
    for sub in ("bulk_cu2o", "bulk_cuo"):
        text = (root / sub / "pw.in").read_text()
        assert "HUBBARD { atomic }" in text
        assert "ortho-atomic" not in text


def test_cli_sweep_analyze_picks_converged_value(tmp_path: Path) -> None:
    """End-to-end: synthesize a sweep tree, then have the CLI pick the converged value."""
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()
    (pseudo_dir / "Cu.upf").write_text("")

    # Generate the sweep tree with the real CLI to keep filename conventions honest.
    runner = CliRunner()
    runner.invoke(
        main,
        [
            "sweep",
            "--param",
            "ecutwfc",
            "--values",
            "40,60,80",
            "--out",
            str(tmp_path / "conv"),
            "--pseudo-dir",
            str(pseudo_dir),
        ],
    )
    # Synthesize pw.out files where energy plateaus by ecutwfc=60 (per-atom).
    for label, energy_ry in (("40", -100.0), ("60", -100.06), ("80", -100.0600001)):
        (tmp_path / "conv" / f"ecutwfc_{label}" / "pw.out").write_text(
            f"!    total energy = {energy_ry} Ry\nJOB DONE.\n"
        )

    result = runner.invoke(
        main,
        ["sweep-analyze", str(tmp_path / "conv"), "--threshold-mev", "10"],
    )
    assert result.exit_code == 0, result.output
    assert "Sweep parameter: ecutwfc" in result.output
    assert "Converged value: ecutwfc = 60" in result.output


def test_cli_sweep_analyze_exits_nonzero_when_not_converged(tmp_path: Path) -> None:
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()
    (pseudo_dir / "Cu.upf").write_text("")

    runner = CliRunner()
    runner.invoke(
        main,
        [
            "sweep",
            "--param",
            "ecutwfc",
            "--values",
            "40,60",
            "--out",
            str(tmp_path / "conv"),
            "--pseudo-dir",
            str(pseudo_dir),
        ],
    )
    # Steep slope between the two points — neither converges at 1 meV/atom.
    for label, energy_ry in (("40", -100.0), ("60", -101.0)):
        (tmp_path / "conv" / f"ecutwfc_{label}" / "pw.out").write_text(
            f"!    total energy = {energy_ry} Ry\nJOB DONE.\n"
        )

    result = runner.invoke(
        main, ["sweep-analyze", str(tmp_path / "conv"), "--threshold-mev", "1"]
    )
    assert result.exit_code != 0
    assert "NONE within threshold" in result.output


def test_cli_aggregate_pourbaix_energies_writes_consumable_json(tmp_path: Path) -> None:
    """Hand-build a make-pourbaix-inputs-shaped tree, aggregate, then feed pourbaix --energies."""
    root = tmp_path / "phase4"
    for sub, energy_ry in (
        ("bulk_cu", -200.0),
        ("bulk_cu2o", -1500.0),  # 6 atoms in conv cell = 2 formula units
        ("bulk_cuo", -2000.0),  # 8 atoms in conv cell = 4 formula units
        ("mol_h2", -2.3),
        ("mol_h2o", -34.0),
    ):
        d = root / sub
        d.mkdir(parents=True)
        (d / "pw.out").write_text(f"!    total energy = {energy_ry} Ry\nJOB DONE.\n")

    runner = CliRunner()
    out_json = tmp_path / "energies.json"
    result = runner.invoke(
        main,
        ["aggregate-pourbaix-energies", str(root), "--out", str(out_json)],
    )
    assert result.exit_code == 0, result.output
    assert out_json.is_file()

    import json as _json

    payload = _json.loads(out_json.read_text())
    names = [p["name"] for p in payload["phases"]]
    assert names == ["Cu(metal)", "Cu2O", "CuO"]
    # Energies in eV (per formula unit). Cu2O total energy was -1500 Ry for 2 f.u.,
    # so per-fu is -750 Ry ≈ -10204.27 eV.
    cu2o_phase = next(p for p in payload["phases"] if p["name"] == "Cu2O")
    assert cu2o_phase["e_dft_ev"] == pytest.approx(-750.0 * 13.605693, rel=1e-5)

    # The JSON must flow straight into `pourbaix --energies` without manual edits.
    follow_up = runner.invoke(
        main,
        ["pourbaix", "--u", "-0.4", "--ph", "7", "--energies", str(out_json)],
    )
    assert follow_up.exit_code == 0, follow_up.output


def test_cli_aggregate_pourbaix_energies_requires_job_done_by_default(
    tmp_path: Path,
) -> None:
    root = tmp_path / "phase4"
    for sub in ("bulk_cu", "bulk_cu2o", "bulk_cuo", "mol_h2", "mol_h2o"):
        d = root / sub
        d.mkdir(parents=True)
        (d / "pw.out").write_text("!    total energy = -100.0 Ry\n")  # no JOB DONE
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "aggregate-pourbaix-energies",
            str(root),
            "--out",
            str(tmp_path / "energies.json"),
        ],
    )
    assert result.exit_code != 0
    assert "did not finish" in result.output


def test_cli_aggregate_pourbaix_energies_allow_incomplete_passes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "phase4"
    for sub in ("bulk_cu", "bulk_cu2o", "bulk_cuo", "mol_h2", "mol_h2o"):
        d = root / sub
        d.mkdir(parents=True)
        (d / "pw.out").write_text("!    total energy = -100.0 Ry\n")
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "aggregate-pourbaix-energies",
            str(root),
            "--out",
            str(tmp_path / "energies.json"),
            "--allow-incomplete",
        ],
    )
    assert result.exit_code == 0, result.output


def test_cli_make_pourbaix_inputs_cu2o_is_nonmagnetic_with_u(tmp_path: Path) -> None:
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()
    for f in ("Cu.upf", "O.upf", "H.upf"):
        (pseudo_dir / f).write_text("")

    root = tmp_path / "phase4"
    runner = CliRunner()
    runner.invoke(
        main,
        ["make-pourbaix-inputs", str(root), "--pseudo-dir", str(pseudo_dir)],
    )
    cu2o_text = (root / "bulk_cu2o" / "pw.in").read_text()
    assert "nspin" in cu2o_text.lower()  # explicitly nspin=1 in our overrides
    # Cu species in the QE 7.1+ HUBBARD card.
    assert "U Cu-3d" in cu2o_text
    # The old &SYSTEM-namelist syntax must NOT appear.
    assert "hubbard_u(1)" not in cu2o_text.lower()
