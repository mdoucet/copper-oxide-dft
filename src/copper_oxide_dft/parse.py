"""Parsers for ``pw.x`` output files.

Lightweight regex-based readers for the quantities a daily-driver
post-mortem needs from each QE calculation:

- Converged total energy and Fermi level.
- Total + absolute magnetization (the AFM-vs-collapsed discriminator —
  total magnetization alone is zero for *both* a correct AFM ground
  state and a magnetism-collapsed run).
- Highest-occupied / lowest-unoccupied levels and the resulting band
  gap, when QE prints them (gapped insulators with smearing, or any
  ``occupations='fixed'`` run).
- Per-site magnetic moments from the high-verbosity "Magnetic moment
  per site" block — load-bearing for confirming AFM ordering on
  CuO-like systems (you expect alternating ~±0.5–0.7 µ_B on the Cu
  sublattices).
- JOB DONE marker.

A derived :meth:`PwResult.magnetic_ordering` label folds the
total/absolute magnetization heuristic so callers don't have to
re-implement it.

Ry-to-eV conversion uses the CODATA value embedded in ASE.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from ase.units import Ry

_TOTAL_ENERGY_RE = re.compile(
    r"^!\s*total energy\s*=\s*([-+0-9.eEdD]+)\s*Ry\s*$", re.MULTILINE
)
_FERMI_RE = re.compile(r"the Fermi energy is\s+([-+0-9.eEdD]+)\s+ev", re.IGNORECASE)
_TOTAL_MAG_RE = re.compile(
    r"^\s*total magnetization\s*=\s*([-+0-9.eEdD]+)\s+Bohr mag/cell\s*$",
    re.MULTILINE,
)
_ABSOLUTE_MAG_RE = re.compile(
    r"^\s*absolute magnetization\s*=\s*([-+0-9.eEdD]+)\s+Bohr mag/cell\s*$",
    re.MULTILINE,
)
_HOMO_LUMO_RE = re.compile(
    r"highest occupied,\s*lowest unoccupied level\s*\(ev\)\s*:\s*"
    r"([-+0-9.eEdD]+)\s+([-+0-9.eEdD]+)",
    re.IGNORECASE,
)
_JOB_DONE_RE = re.compile(r"JOB DONE\.")

# "Magnetic moment per site" block header. QE prints this with
# `verbosity='high'` (our default in write_pw_input) once per
# converged SCF; for vc-relax we want the *last* block.
_SITE_MAG_HEADER_RE = re.compile(
    r"^\s*Magnetic moment per site",
    re.MULTILINE,
)
# One atom line within that block. The (R=...) radius is sometimes
# omitted by older QE versions; both forms must parse.
_SITE_MAG_LINE_RE = re.compile(
    r"^\s*atom\s+\d+\s*(?:\(R=[\d.]+\))?\s*"
    r"charge\s*=\s*[-+0-9.eEdD]+\s*"
    r"magn\s*=\s*([-+0-9.eEdD]+)\s*$",
)

# Heuristic thresholds for magnetic_ordering label. Tuned so a typical
# AFM CuO output (~2-3 µ_B absolute, ~0 total) lands cleanly in "AFM"
# and a magnetism-collapsed CuO (<0.2 µ_B everywhere) lands in
# "non-magnetic".
_MAG_NOISE_THRESHOLD_BOHR = 0.2
"""Absolute magnetization below this counts as non-magnetic. Sized to
sit above SCF noise on a converged run."""

_TOTAL_MAG_AFM_THRESHOLD_BOHR = 0.1
"""``|total mag|`` below this with significant absolute mag = AFM."""

_FM_BALANCE_THRESHOLD_BOHR = 0.1
"""``|absolute - |total||`` below this with significant total mag = FM
(all moments aligned, |total| ≈ absolute)."""


@dataclass(frozen=True)
class PwResult:
    """Parsed quantities from a single ``pw.x`` output file.

    Magnetic fields are ``None`` when QE didn't emit the corresponding
    line (e.g. a non-spin-polarised run has no magnetization at all;
    a low-verbosity run has total/absolute mag but no per-site moments).
    """

    total_energy_ry: float
    total_energy_ev: float
    fermi_energy_ev: float | None
    total_magnetization_bohr: float | None
    absolute_magnetization_bohr: float | None
    homo_ev: float | None
    lumo_ev: float | None
    band_gap_ev: float | None
    site_magnetizations_bohr: tuple[float, ...] | None
    job_done: bool

    @property
    def magnetic_ordering(self) -> str:
        """Heuristic label from total + absolute magnetization.

        Returns one of:

        - ``"unknown"`` — non-magnetic calculation (no absolute mag in
          the output, e.g. ``nspin=1``).
        - ``"non-magnetic"`` — ``nspin=2`` run that collapsed; absolute
          mag ≈ 0. **This is the failure mode for AFM CuO with too-soft
          starting moments / wrong U.**
        - ``"AFM"`` — significant absolute mag, ``|total| ≈ 0``.
        - ``"FM"`` — significant absolute mag, ``|total| ≈ absolute``.
        - ``"ferri/canted"`` — absolute mag and total mag both nonzero
          but not aligned.

        Thresholds are deliberately permissive; see the
        ``_MAG_*_THRESHOLD_BOHR`` module constants.
        """
        if self.absolute_magnetization_bohr is None:
            return "unknown"
        if self.absolute_magnetization_bohr < _MAG_NOISE_THRESHOLD_BOHR:
            return "non-magnetic"
        if self.total_magnetization_bohr is None:
            return "magnetic"
        if abs(self.total_magnetization_bohr) < _TOTAL_MAG_AFM_THRESHOLD_BOHR:
            return "AFM"
        if (
            self.absolute_magnetization_bohr - abs(self.total_magnetization_bohr)
            < _FM_BALANCE_THRESHOLD_BOHR
        ):
            return "FM"
        return "ferri/canted"


def _parse_float_qe(token: str) -> float:
    # QE sometimes emits Fortran-style "d" exponents; Python wants "e".
    return float(token.replace("D", "e").replace("d", "e"))


def _last_float(pattern: re.Pattern[str], text: str) -> float | None:
    matches = pattern.findall(text)
    if not matches:
        return None
    return _parse_float_qe(matches[-1])


def _last_homo_lumo(text: str) -> tuple[float | None, float | None]:
    matches = _HOMO_LUMO_RE.findall(text)
    if not matches:
        return None, None
    homo, lumo = matches[-1]
    return _parse_float_qe(homo), _parse_float_qe(lumo)


def _last_site_magnetizations(text: str) -> tuple[float, ...] | None:
    """Pull the final ``Magnetic moment per site`` block's per-atom moments.

    Returns ``None`` if no such block exists (low-verbosity run or
    nspin=1). Returns an empty tuple if a header is present but no
    parsable atom lines follow — unusual but flagged distinctly from
    "block not present" so it surfaces a malformed output.
    """
    headers = list(_SITE_MAG_HEADER_RE.finditer(text))
    if not headers:
        return None
    block = text[headers[-1].end() :]
    mags: list[float] = []
    for line in block.splitlines():
        match = _SITE_MAG_LINE_RE.match(line)
        if match:
            mags.append(_parse_float_qe(match.group(1)))
            continue
        if mags:
            # First non-matching line after we started accumulating
            # marks the end of the block.
            break
    return tuple(mags)


def parse_pw_output(path: str | os.PathLike[str]) -> PwResult:
    """Parse a ``pw.x`` standard-output file.

    Only the final converged values are returned, even if multiple SCF
    cycles ran (vc-relax, relax). The energy is read from lines prefixed
    by ``!`` which QE uses to mark converged totals.

    Args:
        path: Path to the captured stdout of a ``pw.x`` run.

    Returns:
        Parsed scalar quantities. Fields that QE didn't emit (no
        magnetism, no gap, low verbosity) come back as ``None``.

    Raises:
        ValueError: If no converged total-energy line is found.
    """
    text = Path(path).read_text()
    total_energy_ry = _last_float(_TOTAL_ENERGY_RE, text)
    if total_energy_ry is None:
        raise ValueError(
            f"No converged total-energy line ('! total energy = ...') found in {path}"
        )

    homo_ev, lumo_ev = _last_homo_lumo(text)
    band_gap_ev = (
        (lumo_ev - homo_ev) if (homo_ev is not None and lumo_ev is not None) else None
    )

    return PwResult(
        total_energy_ry=total_energy_ry,
        total_energy_ev=total_energy_ry * Ry,
        fermi_energy_ev=_last_float(_FERMI_RE, text),
        total_magnetization_bohr=_last_float(_TOTAL_MAG_RE, text),
        absolute_magnetization_bohr=_last_float(_ABSOLUTE_MAG_RE, text),
        homo_ev=homo_ev,
        lumo_ev=lumo_ev,
        band_gap_ev=band_gap_ev,
        site_magnetizations_bohr=_last_site_magnetizations(text),
        job_done=bool(_JOB_DONE_RE.search(text)),
    )
