"""Nudged Elastic Band (NEB) input generation for QE's ``neb.x``.

Phase 8 scaffold for reconstruction-barrier studies. QE's NEB binary
(``neb.x``, distinct from ``pw.x``) takes a single input file with:

* A ``&PATH`` namelist controlling the NEB algorithm.
* A repeated block of ``BEGIN_IMAGE`` / ``END_IMAGE`` sections, each of
  which is essentially a regular ``pw.x`` input describing one image
  along the reaction coordinate. The first and last images are pinned
  (initial and final states); intermediate images are interpolated.

This module wraps that format. A typical use is:

    initial = build_cu111_slab() + adsorbate at site A
    final   = build_cu111_slab() + adsorbate at site B
    write_neb_input(
        out_path="run/neb/neb.in",
        endpoints=(initial, final),
        n_intermediate_images=5,
        pseudopotentials={"Cu": "Cu.upf", "O": "O.upf"},
        pseudo_dir="...",
    )

Note: the actual minimum-energy-path *interpolation* between images is
QE's job (``neb.x`` builds intermediate images from the two endpoints
when ``num_of_images`` exceeds the supplied image count). This writer
just emits the endpoints plus the namelists.

Reference:
    https://www.quantum-espresso.org/Doc/INPUT_NEB.html
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.io.espresso import write_espresso_in

from copper_oxide_dft.qe_input import (
    DEFAULT_DEGAUSS_RY,
    DEFAULT_ECUTWFC_RY,
    _format_namelist_value,
    _resolve_pseudo_dir,
)


def write_neb_input(
    out_path: str | os.PathLike[str],
    *,
    endpoints: tuple[Atoms, Atoms],
    n_intermediate_images: int,
    pseudopotentials: Mapping[str, str],
    pseudo_dir: str | os.PathLike[str] | None = None,
    prefix: str = "neb",
    ecutwfc: float = DEFAULT_ECUTWFC_RY,
    ecutrho: float | None = None,
    kpts: tuple[int, int, int] = (4, 4, 1),
    degauss: float = DEFAULT_DEGAUSS_RY,
    ci_scheme: str = "auto",
    k_min: float = 0.1,
    k_max: float = 0.3,
    nstep_path: int = 100,
    path_thr_ev_per_a: float = 0.05,
    extra_path: Mapping[str, Any] | None = None,
    extra_pw_input: Mapping[str, Mapping[str, Any]] | None = None,
) -> Path:
    """Write a QE ``neb.x`` input file with two pinned endpoints.

    Args:
        out_path: Path to write the input to.
        endpoints: ``(initial, final)`` ASE structures. Must have the
            same chemical composition and cell.
        n_intermediate_images: Number of NEB images BETWEEN the
            endpoints. Total ``num_of_images`` = ``n_intermediate + 2``.
        pseudopotentials: Symbol -> UPF filename mapping.
        pseudo_dir: Pseudo directory. Falls back to
            ``$CUOXDFT_PSEUDO_DIR`` if ``None``.
        prefix: QE ``prefix`` for the NEB run.
        ecutwfc / ecutrho / kpts / degauss: Standard pw.x cutoffs and
            grid. Applied identically to every image.
        ci_scheme: ``"auto"`` (climbing image switches on automatically)
            or ``"no-CI"`` (plain NEB; only converges to a saddle-point
            estimate, not the exact saddle).
        k_min / k_max: Spring-constant bounds (Ry/Bohr) along the path.
        nstep_path: Max number of NEB iterations.
        path_thr_ev_per_a: Convergence threshold on the perpendicular
            force component (eV/Å).
        extra_path: Optional overrides merged into the ``&PATH``
            namelist.
        extra_pw_input: Optional overrides merged into the per-image
            ``pw.x`` namelists (e.g. spin polarization for AFM systems).

    Returns:
        Path to the written input file.

    Raises:
        ValueError: If the endpoints have different atom counts or
            chemical formulas, or if ``n_intermediate_images < 1``.
    """
    if n_intermediate_images < 1:
        raise ValueError(
            f"n_intermediate_images must be >= 1; got {n_intermediate_images}."
        )
    initial, final = endpoints
    if len(initial) != len(final):
        raise ValueError(
            f"Endpoint atom counts differ: {len(initial)} vs {len(final)}."
        )
    if initial.get_chemical_formula() != final.get_chemical_formula():
        raise ValueError(
            "Endpoints have different chemical formulas: "
            f"{initial.get_chemical_formula()} vs {final.get_chemical_formula()}."
        )

    pseudo_resolved = _resolve_pseudo_dir(pseudo_dir)
    if ecutrho is None:
        ecutrho = 8.0 * ecutwfc
    num_of_images = n_intermediate_images + 2

    path_namelist: dict[str, Any] = {
        "string_method": "neb",
        "restart_mode": "from_scratch",
        "nstep_path": nstep_path,
        "ds": 1.0,
        "opt_scheme": "broyden",
        "num_of_images": num_of_images,
        "k_max": k_max,
        "k_min": k_min,
        "CI_scheme": ci_scheme,
        "path_thr": path_thr_ev_per_a,
    }
    if extra_path:
        path_namelist.update(extra_path)

    pw_input: dict[str, dict[str, Any]] = {
        "control": {
            "prefix": prefix,
            "pseudo_dir": str(pseudo_resolved),
            "outdir": "./tmp",
            "tstress": True,
            "tprnfor": True,
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
    if extra_pw_input:
        for namelist, entries in extra_pw_input.items():
            pw_input.setdefault(namelist, {}).update(entries)

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    parts: list[str] = ["BEGIN", "BEGIN_PATH_INPUT", "&PATH"]
    for key, value in path_namelist.items():
        parts.append(f"  {key} = {_format_namelist_value(value)}")
    parts.append("/")
    parts.append("END_PATH_INPUT")
    parts.append("BEGIN_ENGINE_INPUT")

    # Use ASE's namelist writer for the engine block; it knows how to
    # emit ATOMIC_SPECIES / K_POINTS in QE's exact whitespace.
    from io import StringIO

    engine_buffer = StringIO()
    write_espresso_in(
        engine_buffer,
        initial,
        input_data=pw_input,
        pseudopotentials=dict(pseudopotentials),
        kpts=kpts,
    )
    engine_text = engine_buffer.getvalue()
    parts.append(engine_text.rstrip())

    parts.append("BEGIN_POSITIONS")
    parts.append("FIRST_IMAGE")
    parts.append(_format_positions(initial))
    parts.append("LAST_IMAGE")
    parts.append(_format_positions(final))
    parts.append("END_POSITIONS")
    parts.append("END_ENGINE_INPUT")
    parts.append("END")

    p.write_text("\n".join(parts) + "\n")
    return p


def _format_positions(atoms: Atoms) -> str:
    """Render ATOMIC_POSITIONS in QE's alat/angstrom 'angstrom' format."""
    lines = ["ATOMIC_POSITIONS angstrom"]
    for atom in atoms:
        x, y, z = atom.position
        lines.append(f"  {atom.symbol}  {x:.8f}  {y:.8f}  {z:.8f}")
    return "\n".join(lines)


__all__ = ("write_neb_input",)
