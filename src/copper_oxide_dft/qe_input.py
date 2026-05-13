"""Quantum ESPRESSO ``pw.x`` input file generation with project defaults.

Wraps :func:`ase.io.espresso.write_espresso_in` with the project's standard
choices (Marzari–Vanderbilt smearing, PBE+U-ready namelist structure,
sensible SCF convergence thresholds). Later phases extend this with relax,
vc-relax, and Hubbard-U-aware variants.

Pseudopotentials are looked up relative to ``pseudo_dir``. If not passed
explicitly, the environment variable ``CUOXDFT_PSEUDO_DIR`` is used; the
function raises :class:`FileNotFoundError` if neither resolves to an
existing directory.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.io.espresso import write_espresso_in

PSEUDO_DIR_ENV_VAR = "CUOXDFT_PSEUDO_DIR"

DEFAULT_ECUTWFC_RY = 80.0
"""Wavefunction plane-wave cutoff. Conservative starting point; refined in
the Phase 1 convergence sweep."""

DEFAULT_DEGAUSS_RY = 0.02
"""Smearing width for Marzari–Vanderbilt cold smearing. Required for metallic Cu;
see docs/ground_truths.md (Cu oxide DFT gotchas)."""


def _resolve_pseudo_dir(pseudo_dir: str | os.PathLike[str] | None) -> Path:
    if pseudo_dir is None:
        env = os.environ.get(PSEUDO_DIR_ENV_VAR)
        if not env:
            raise FileNotFoundError(
                f"Pseudopotential directory not provided and ${PSEUDO_DIR_ENV_VAR} is unset. "
                "Pass `pseudo_dir=` or set the environment variable."
            )
        pseudo_dir = env
    resolved = Path(pseudo_dir).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Pseudopotential directory does not exist: {resolved}")
    return resolved


def write_scf_input(
    atoms: Atoms,
    out_path: str | os.PathLike[str],
    pseudopotentials: Mapping[str, str],
    *,
    prefix: str = "calc",
    ecutwfc: float = DEFAULT_ECUTWFC_RY,
    ecutrho: float | None = None,
    kpts: tuple[int, int, int] = (8, 8, 8),
    koffset: tuple[int, int, int] = (0, 0, 0),
    degauss: float = DEFAULT_DEGAUSS_RY,
    pseudo_dir: str | os.PathLike[str] | None = None,
    extra_input_data: Mapping[str, Mapping[str, Any]] | None = None,
) -> Path:
    """Write a ``pw.x`` SCF input file with project-standard defaults.

    Args:
        atoms: Structure to compute. Cell and positions are written directly;
            ``ibrav=0`` is used so QE consumes the supplied vectors.
        out_path: File path to write the input to. Parent directories are
            created if needed.
        pseudopotentials: Mapping from chemical symbol to UPF filename
            (e.g. ``{"Cu": "Cu.upf"}``). Files must live in ``pseudo_dir``.
        prefix: ``CONTROL.prefix`` value (used by QE for output filenames).
        ecutwfc: Plane-wave wavefunction cutoff (Ry).
        ecutrho: Charge-density cutoff (Ry). Defaults to ``8 * ecutwfc``,
            which is appropriate for PAW pseudopotentials.
        kpts: Monkhorst–Pack k-point grid.
        koffset: Grid offset (use ``(1, 1, 1)`` for shifted, Γ-excluded).
        degauss: Smearing width (Ry).
        pseudo_dir: Directory containing UPF files. Falls back to
            ``$CUOXDFT_PSEUDO_DIR`` if ``None``.
        extra_input_data: Optional namelist overrides merged on top of
            defaults (e.g. ``{"system": {"nspin": 2}}``).

    Returns:
        Path to the written input file.

    Raises:
        FileNotFoundError: If ``pseudo_dir`` and ``$CUOXDFT_PSEUDO_DIR`` are
            both missing or do not point to an existing directory.
    """
    pseudo_dir_resolved = _resolve_pseudo_dir(pseudo_dir)

    if ecutrho is None:
        ecutrho = 8.0 * ecutwfc

    input_data: dict[str, dict[str, Any]] = {
        "control": {
            "calculation": "scf",
            "prefix": prefix,
            "pseudo_dir": str(pseudo_dir_resolved),
            "outdir": "./tmp",
            "tstress": True,
            "tprnfor": True,
            "verbosity": "high",
        },
        "system": {
            "ibrav": 0,
            "ecutwfc": ecutwfc,
            "ecutrho": ecutrho,
            "occupations": "smearing",
            "smearing": "mv",
            "degauss": degauss,
        },
        "electrons": {
            "conv_thr": 1.0e-8,
            "mixing_beta": 0.4,
            "electron_maxstep": 200,
        },
    }

    if extra_input_data:
        for namelist, overrides in extra_input_data.items():
            input_data.setdefault(namelist, {}).update(overrides)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        write_espresso_in(
            fh,
            atoms,
            input_data=input_data,
            pseudopotentials=dict(pseudopotentials),
            kpts=kpts,
            koffset=koffset,
        )
    return out_path
