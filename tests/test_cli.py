"""Tests for the copper-oxide-dft CLI."""

from __future__ import annotations

from pathlib import Path

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
