"""Tests for copper_oxide_dft.parse.

Uses small synthetic pw.x-style stdout fixtures rather than real QE
output to keep tests fast and hermetic.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from ase.units import Ry

from copper_oxide_dft.parse import parse_pw_output

# Minimal scf-converged stdout (non-magnetic).
SCF_OUTPUT = """\
     Self-consistent Calculation
     iteration #  1     ecut=    60.00 Ry
     ...
     the Fermi energy is    12.3456 ev
     ...
!    total energy              =   -1234.56789012 Ry
     ...
     total magnetization       =     0.00 Bohr mag/cell
     absolute magnetization    =     0.00 Bohr mag/cell

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

# AFM CuO output (final SCF converged): total magnetization cancels but
# absolute mag is significant, alternating per-site moments, gap ~1.3 eV.
AFM_CUO_OUTPUT = """\
     Self-consistent Calculation
     ...
     iteration # 12     ecut=   100.00 Ry
     ...
     Magnetic moment per site  (integrated on atomic sphere of radius R)
     atom   1 (R=0.357)  charge=  9.7892  magn= 0.6541
     atom   2 (R=0.357)  charge=  9.7891  magn=-0.6541
     atom   3 (R=0.357)  charge=  9.7891  magn= 0.6541
     atom   4 (R=0.357)  charge=  9.7891  magn=-0.6541
     atom   5 (R=0.117)  charge=  3.7654  magn= 0.0001
     atom   6 (R=0.117)  charge=  3.7654  magn= 0.0001
     atom   7 (R=0.117)  charge=  3.7654  magn= 0.0001
     atom   8 (R=0.117)  charge=  3.7654  magn= 0.0001

     total magnetization       =     0.00 Bohr mag/cell
     absolute magnetization    =     2.62 Bohr mag/cell

     highest occupied, lowest unoccupied level (ev):    -0.7234   0.5678

     the Fermi energy is     -0.0712 ev
!    total energy              =   -1648.65002800 Ry
   JOB DONE.
"""

# Magnetism-collapsed CuO: nspin=2 ran but the AFM solution wasn't
# preserved — total *and* absolute mag essentially zero, no per-site
# moments emitted (the high-verbosity block still prints but all magn
# columns are noise), no gap.
COLLAPSED_CUO_OUTPUT = """\
     Self-consistent Calculation
     ...
     Magnetic moment per site  (integrated on atomic sphere of radius R)
     atom   1 (R=0.357)  charge=  9.7892  magn= 0.0023
     atom   2 (R=0.357)  charge=  9.7892  magn=-0.0021
     atom   3 (R=0.357)  charge=  9.7892  magn= 0.0019
     atom   4 (R=0.357)  charge=  9.7892  magn=-0.0017

     total magnetization       =     0.00 Bohr mag/cell
     absolute magnetization    =     0.01 Bohr mag/cell

     the Fermi energy is     10.5868 ev
!    total energy              =   -1648.10000000 Ry
   JOB DONE.
"""

# Ferromagnetic-ish: total mag ~= absolute mag (all aligned).
FM_OUTPUT = """\
!    total energy              =   -500.0 Ry
     total magnetization       =     4.00 Bohr mag/cell
     absolute magnetization    =     4.05 Bohr mag/cell
     the Fermi energy is     7.5 ev
   JOB DONE.
"""

# Metallic with smearing: no HOMO/LUMO line emitted.
METALLIC_OUTPUT = """\
!    total energy              =   -800.0 Ry
     the Fermi energy is     8.5 ev
     total magnetization       =     0.00 Bohr mag/cell
     absolute magnetization    =     0.00 Bohr mag/cell
   JOB DONE.
"""

# Older QE site-magnetization formatting without the (R=...) field.
LEGACY_SITE_MAG_OUTPUT = """\
!    total energy              =   -1000.0 Ry
     Magnetic moment per site
     atom   1   charge=  9.8000  magn=  0.5000
     atom   2   charge=  9.8000  magn= -0.5000
     total magnetization       =     0.00 Bohr mag/cell
     absolute magnetization    =     1.00 Bohr mag/cell
   JOB DONE.
"""


def test_parse_scf_output_extracts_all_fields(tmp_path: Path) -> None:
    path = tmp_path / "scf.out"
    path.write_text(SCF_OUTPUT)

    result = parse_pw_output(path)

    assert result.total_energy_ry == pytest.approx(-1234.56789012)
    assert result.total_energy_ev == pytest.approx(-1234.56789012 * Ry)
    assert result.fermi_energy_ev == pytest.approx(12.3456)
    assert result.total_magnetization_bohr == pytest.approx(0.0)
    assert result.absolute_magnetization_bohr == pytest.approx(0.0)
    assert result.homo_ev is None
    assert result.lumo_ev is None
    assert result.band_gap_ev is None
    assert result.site_magnetizations_bohr is None
    assert result.job_done is True


def test_parse_returns_last_converged_energy_for_vc_relax(tmp_path: Path) -> None:
    path = tmp_path / "vc.out"
    path.write_text(VC_RELAX_OUTPUT)

    result = parse_pw_output(path)

    assert result.total_energy_ry == pytest.approx(-1001.5)
    assert result.fermi_energy_ev == pytest.approx(5.6)
    assert result.total_magnetization_bohr is None  # not emitted in this fixture
    assert result.absolute_magnetization_bohr is None
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


# ---- AFM-converged CuO ------------------------------------------------------


def test_parse_afm_cuo_extracts_absolute_magnetization(tmp_path: Path) -> None:
    """The discriminator field — total mag is 0 for AFM AND for collapsed."""
    path = tmp_path / "afm.out"
    path.write_text(AFM_CUO_OUTPUT)

    result = parse_pw_output(path)

    assert result.total_magnetization_bohr == pytest.approx(0.0)
    assert result.absolute_magnetization_bohr == pytest.approx(2.62)


def test_parse_afm_cuo_extracts_per_site_moments(tmp_path: Path) -> None:
    """High-verbosity 'Magnetic moment per site' block parses to a tuple."""
    path = tmp_path / "afm.out"
    path.write_text(AFM_CUO_OUTPUT)

    result = parse_pw_output(path)

    assert result.site_magnetizations_bohr is not None
    assert len(result.site_magnetizations_bohr) == 8
    # Alternating Cu moments — load-bearing for "AFM survived" verdict.
    assert result.site_magnetizations_bohr[0] == pytest.approx(0.6541)
    assert result.site_magnetizations_bohr[1] == pytest.approx(-0.6541)
    # O sites near zero.
    assert abs(result.site_magnetizations_bohr[-1]) < 0.01


def test_parse_afm_cuo_extracts_band_gap(tmp_path: Path) -> None:
    path = tmp_path / "afm.out"
    path.write_text(AFM_CUO_OUTPUT)

    result = parse_pw_output(path)

    assert result.homo_ev == pytest.approx(-0.7234)
    assert result.lumo_ev == pytest.approx(0.5678)
    assert result.band_gap_ev == pytest.approx(0.5678 - (-0.7234))


def test_parse_afm_cuo_magnetic_ordering_is_afm(tmp_path: Path) -> None:
    """Heuristic label folds the total + absolute mag check."""
    path = tmp_path / "afm.out"
    path.write_text(AFM_CUO_OUTPUT)

    result = parse_pw_output(path)

    assert result.magnetic_ordering == "AFM"


# ---- Collapsed magnetism ----------------------------------------------------


def test_parse_collapsed_cuo_magnetic_ordering_is_non_magnetic(tmp_path: Path) -> None:
    """Absolute mag near zero with nspin=2 means AFM was lost — flag it."""
    path = tmp_path / "collapsed.out"
    path.write_text(COLLAPSED_CUO_OUTPUT)

    result = parse_pw_output(path)

    assert result.total_magnetization_bohr == pytest.approx(0.0)
    assert result.absolute_magnetization_bohr == pytest.approx(0.01)
    # No HOMO/LUMO line in this output — band_gap stays None even though
    # the run technically "converged" (gap should have shown up if AFM
    # survived; its absence is the smoking gun together with abs_mag≈0).
    assert result.band_gap_ev is None
    assert result.magnetic_ordering == "non-magnetic"


# ---- Ferromagnetic ----------------------------------------------------------


def test_parse_fm_output_magnetic_ordering_is_fm(tmp_path: Path) -> None:
    """|total| ≈ absolute → moments aligned → FM."""
    path = tmp_path / "fm.out"
    path.write_text(FM_OUTPUT)

    result = parse_pw_output(path)

    assert result.magnetic_ordering == "FM"


# ---- Metallic ---------------------------------------------------------------


def test_parse_metallic_output_has_no_band_gap(tmp_path: Path) -> None:
    """Smearing-only output produces no HOMO/LUMO line."""
    path = tmp_path / "metal.out"
    path.write_text(METALLIC_OUTPUT)

    result = parse_pw_output(path)

    assert result.homo_ev is None
    assert result.lumo_ev is None
    assert result.band_gap_ev is None
    assert result.fermi_energy_ev == pytest.approx(8.5)


def test_parse_metallic_output_magnetic_ordering_is_non_magnetic(tmp_path: Path) -> None:
    path = tmp_path / "metal.out"
    path.write_text(METALLIC_OUTPUT)

    result = parse_pw_output(path)

    assert result.magnetic_ordering == "non-magnetic"


# ---- Legacy QE site-mag format ---------------------------------------------


def test_parse_legacy_site_mag_format_without_radius(tmp_path: Path) -> None:
    """Older QE versions omit the (R=...) radius — must still parse."""
    path = tmp_path / "legacy.out"
    path.write_text(LEGACY_SITE_MAG_OUTPUT)

    result = parse_pw_output(path)

    assert result.site_magnetizations_bohr == (0.5, -0.5)


# ---- vc-relax with multiple "Magnetic moment per site" blocks ---------------


def test_parse_picks_last_site_mag_block_in_vc_relax(tmp_path: Path) -> None:
    """vc-relax emits the block per BFGS step; we want the final one."""
    two_block_output = """\
     iteration #  1
     Magnetic moment per site
     atom   1   charge=  9.8  magn=  0.30
     atom   2   charge=  9.8  magn= -0.30

!    total energy              =   -1000.0 Ry
     iteration #  2
     Magnetic moment per site
     atom   1   charge=  9.8  magn=  0.65
     atom   2   charge=  9.8  magn= -0.65

!    total energy              =   -1001.0 Ry
     total magnetization       =     0.00 Bohr mag/cell
     absolute magnetization    =     1.30 Bohr mag/cell
   JOB DONE.
"""
    path = tmp_path / "vc_multi.out"
    path.write_text(two_block_output)

    result = parse_pw_output(path)

    # Last block: ±0.65, not ±0.30.
    assert result.site_magnetizations_bohr == (0.65, -0.65)
    assert result.total_energy_ry == pytest.approx(-1001.0)


# ---- Magnetic-ordering label edge cases -------------------------------------


def test_magnetic_ordering_unknown_when_no_absolute_mag(tmp_path: Path) -> None:
    """nspin=1 calc: no absolute mag line at all → can't classify."""
    path = tmp_path / "ns1.out"
    path.write_text(VC_RELAX_OUTPUT)  # has no mag lines

    result = parse_pw_output(path)

    assert result.magnetic_ordering == "unknown"
