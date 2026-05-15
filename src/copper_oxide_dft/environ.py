"""Environ implicit-solvation input file generation (Phase 5 scaffold).

QE+Environ runs require a second input file (``environ.in``) alongside
the standard ``pw.in``. Environ reads this file at runtime when the
binary is the Environ-patched build of ``pw.x``. The patch is not in
the stock ORNL QE module (verify before use); see ground_truths.md for
how to build it locally on Frontier.

The defaults below pick the "implicit water at vacuum/solid interface"
preset that is the standard starting point for solvated electrochemistry:

* ``environ_type = 'water'``: use the built-in water parameter set
  (ε_static = 78.36, surface tension and pressure terms zeroed out by
  default — we use a purely electrostatic solvent for Phase 5).
* ``solvent_mode = 'electronic'``: build the dielectric cavity from the
  self-consistent electron density (Andreussi soft-sphere algorithm),
  which is the standard for slabs.
* ``pbc_correction = 'parabolic'`` with ``pbc_dim = 2``: 2D periodic
  boundary correction so the implicit-water side does not couple to its
  own periodic image across the vacuum.

Phase 5 only needs these knobs to land; finer-grained options (ionic
electrolyte concentrations, custom dielectric profiles) can be passed
via ``extra_namelists`` once we get there.

References:
    https://environ.readthedocs.io/en/latest/
    Andreussi et al., J. Chem. Phys. 136, 064102 (2012).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from copper_oxide_dft.qe_input import _format_namelist_value

WATER_STATIC_PERMITTIVITY = 78.36
"""Static dielectric constant of liquid water at 298 K (Environ default)."""


def write_environ_input(
    out_path: str | os.PathLike[str],
    *,
    environ_type: str = "water",
    static_permittivity: float = WATER_STATIC_PERMITTIVITY,
    solvent_mode: str = "electronic",
    cavity_alpha: float = 1.12,
    pbc_dim: int = 2,
    pbc_axis: int = 3,
    pbc_correction: str = "parabolic",
    verbosity: int = 0,
    extra_namelists: Mapping[str, Mapping[str, Any]] | None = None,
) -> Path:
    """Write an ``environ.in`` next to a ``pw.in`` for Environ-patched QE.

    Args:
        out_path: Destination path (typically ``<dir>/environ.in``).
        environ_type: Environ solvent preset. ``"water"`` picks the
            built-in liquid-water parameter set; ``"vacuum"`` is a
            no-solvent dry-run useful for differential tests.
        static_permittivity: Override for the static dielectric constant.
            Most users keep the ``"water"`` preset default.
        solvent_mode: How the dielectric cavity is built. ``"electronic"``
            uses the self-consistent density (Andreussi); ``"ionic"`` is
            available but rarely used for slabs.
        cavity_alpha: Cavity scaling factor for the soft-sphere
            algorithm. 1.12 is the Environ default and works for most
            transition-metal surfaces.
        pbc_dim: Dimensionality of the periodic-boundary correction.
            2 = slab; 3 = bulk (not appropriate for solvated calcs).
        pbc_axis: Cartesian axis normal to the slab (1, 2, or 3).
        pbc_correction: PBC correction method (``parabolic`` or ``ms``).
        verbosity: Environ verbosity level (0 = quiet).
        extra_namelists: Optional dict-of-dicts merged over the defaults
            (e.g. ``{"electrostatic": {"tolvelect": 1.0e-12}}``).

    Returns:
        Path to the written ``environ.in``.
    """
    namelists: dict[str, dict[str, Any]] = {
        "environ": {
            "verbose": verbosity,
            "environ_thr": 1.0e-2,
            "environ_type": environ_type,
            "env_static_permittivity": static_permittivity,
            "env_electrostatic": True,
            "env_surface_tension": 0.0,
            "env_pressure": 0.0,
        },
        "boundary": {
            "solvent_mode": solvent_mode,
            "alpha": cavity_alpha,
        },
        "electrostatic": {
            "pbc_correction": pbc_correction,
            "pbc_dim": pbc_dim,
            "pbc_axis": pbc_axis,
            "tolvelect": 5.0e-13,
        },
    }
    if extra_namelists:
        for name, entries in extra_namelists.items():
            namelists.setdefault(name, {}).update(entries)

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for name, entries in namelists.items():
        lines.append(f"&{name.upper()}")
        for key, value in entries.items():
            lines.append(f"  {key} = {_format_namelist_value(value)}")
        lines.append("/")
        lines.append("")  # blank line between namelists for readability
    p.write_text("\n".join(lines).rstrip() + "\n")
    return p


__all__ = ("WATER_STATIC_PERMITTIVITY", "write_environ_input")
