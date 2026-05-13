"""Parsers for ``pw.x`` output files.

Lightweight regex-based readers for the few quantities we need from each
QE calculation: converged total energy, Fermi energy, total magnetization,
and whether the SCF cycle converged at all.

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
_MAGNETIZATION_RE = re.compile(
    r"^\s*total magnetization\s*=\s*([-+0-9.eEdD]+)\s+Bohr mag/cell\s*$",
    re.MULTILINE,
)
_JOB_DONE_RE = re.compile(r"JOB DONE\.")


@dataclass(frozen=True)
class PwResult:
    """Parsed quantities from a single ``pw.x`` output file."""

    total_energy_ry: float
    total_energy_ev: float
    fermi_energy_ev: float | None
    total_magnetization_bohr: float | None
    job_done: bool


def _parse_float_qe(token: str) -> float:
    # QE sometimes emits Fortran-style "d" exponents; Python wants "e".
    return float(token.replace("D", "e").replace("d", "e"))


def _last_float(pattern: re.Pattern[str], text: str) -> float | None:
    matches = pattern.findall(text)
    if not matches:
        return None
    return _parse_float_qe(matches[-1])


def parse_pw_output(path: str | os.PathLike[str]) -> PwResult:
    """Parse a ``pw.x`` standard-output file.

    Only the final converged values are returned, even if multiple SCF
    cycles ran (vc-relax, relax). The energy is read from lines prefixed
    by ``!`` which QE uses to mark converged totals.

    Args:
        path: Path to the captured stdout of a ``pw.x`` run.

    Returns:
        Parsed scalar quantities.

    Raises:
        ValueError: If no converged total-energy line is found.
    """
    text = Path(path).read_text()
    total_energy_ry = _last_float(_TOTAL_ENERGY_RE, text)
    if total_energy_ry is None:
        raise ValueError(
            f"No converged total-energy line ('! total energy = ...') found in {path}"
        )
    return PwResult(
        total_energy_ry=total_energy_ry,
        total_energy_ev=total_energy_ry * Ry,
        fermi_energy_ev=_last_float(_FERMI_RE, text),
        total_magnetization_bohr=_last_float(_MAGNETIZATION_RE, text),
        job_done=bool(_JOB_DONE_RE.search(text)),
    )
