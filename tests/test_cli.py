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
