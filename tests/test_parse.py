"""Tests for copper_oxide_dft.parse.

Uses small synthetic pw.x-style stdout fixtures rather than real QE
output to keep tests fast and hermetic.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from ase.units import Ry

from copper_oxide_dft.parse import parse_pw_output

# Minimal scf-converged stdout.
SCF_OUTPUT = """\
     Self-consistent Calculation
     iteration #  1     ecut=    60.00 Ry
     ...
     the Fermi energy is    12.3456 ev
     ...
!    total energy              =   -1234.56789012 Ry
     ...
     total magnetization       =     0.00 Bohr mag/cell

     ...
   JOB DONE.
"""

VC_RELAX_OUTPUT = """\
     vc-relax: BFGS step 1
!    total energy              =   -1000.0000 Ry
     the Fermi energy is     5.5 ev
     vc-relax: BFGS step 2
!    total energy              =   -1001.5000 Ry
     the Fermi energy is     5.6 ev
   JOB DONE.
"""

UNCONVERGED_OUTPUT = """\
     Self-consistent Calculation
     iteration #  50  ecut=  60.00 Ry
     convergence NOT achieved
"""


def test_parse_scf_output_extracts_all_fields(tmp_path: Path) -> None:
    path = tmp_path / "scf.out"
    path.write_text(SCF_OUTPUT)

    result = parse_pw_output(path)

    assert result.total_energy_ry == pytest.approx(-1234.56789012)
    assert result.total_energy_ev == pytest.approx(-1234.56789012 * Ry)
    assert result.fermi_energy_ev == pytest.approx(12.3456)
    assert result.total_magnetization_bohr == pytest.approx(0.0)
    assert result.job_done is True


def test_parse_returns_last_converged_energy_for_vc_relax(tmp_path: Path) -> None:
    path = tmp_path / "vc.out"
    path.write_text(VC_RELAX_OUTPUT)

    result = parse_pw_output(path)

    assert result.total_energy_ry == pytest.approx(-1001.5)
    assert result.fermi_energy_ev == pytest.approx(5.6)
    assert result.total_magnetization_bohr is None  # not emitted in this fixture
    assert result.job_done is True


def test_parse_raises_when_no_converged_energy(tmp_path: Path) -> None:
    path = tmp_path / "unconv.out"
    path.write_text(UNCONVERGED_OUTPUT)

    with pytest.raises(ValueError, match="No converged total-energy"):
        parse_pw_output(path)


def test_parse_marks_not_done_when_job_done_missing(tmp_path: Path) -> None:
    path = tmp_path / "running.out"
    path.write_text(SCF_OUTPUT.replace("JOB DONE.", ""))

    result = parse_pw_output(path)

    assert result.job_done is False
