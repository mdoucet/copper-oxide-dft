"""Convergence-sweep helpers for the Phase 1 bulk-Cu calibration.

Generates a directory tree of ``pw.x`` inputs varying one parameter at a
time (``ecutwfc``, k-point grid size, or smearing width). Outputs land at
``<out_root>/<param>_<value>/pw.in`` so the user can submit each one as a
separate job on the cluster and aggregate results with
:mod:`copper_oxide_dft.parse`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ase import Atoms

from copper_oxide_dft.qe_input import spin_and_hubbard_overrides, write_pw_input

SUPPORTED_SWEEP_PARAMETERS = frozenset({"ecutwfc", "kpts", "degauss", "hubbard_u"})


def _format_value(param: str, value: float | int) -> str:
    if param == "ecutwfc":
        return f"{value:.0f}"
    if param == "kpts":
        return f"{int(value)}"
    if param == "hubbard_u":
        # U values are small floats (0-10 eV typically); two decimals are
        # enough and keep directory names short. "1p00" reads cleanly.
        return f"{value:.2f}".replace(".", "p")
    # param == "degauss" (validated upstream in sweep_convergence)
    return f"{value:.3f}".replace(".", "p")


def sweep_convergence(
    atoms: Atoms,
    out_root: str | os.PathLike[str],
    pseudopotentials: Mapping[str, str],
    *,
    param: str,
    values: Sequence[float] | Sequence[int],
    pseudo_dir: str | os.PathLike[str] | None = None,
    base_kwargs: Mapping[str, Any] | None = None,
) -> list[Path]:
    """Generate one ``pw.x`` SCF input per value of a sweep parameter.

    Args:
        atoms: Structure used for every point in the sweep.
        out_root: Directory tree root. Each sweep point lives in
            ``<out_root>/<param>_<formatted_value>/``.
        pseudopotentials: UPF filename mapping, forwarded to
            :func:`write_pw_input`.
        param: Parameter being varied. One of
            :data:`SUPPORTED_SWEEP_PARAMETERS`.
        values: Sweep values. For ``param="kpts"`` each value ``n`` is
            expanded to a Monkhorst-Pack grid ``(n, n, n)``.
        pseudo_dir: Forwarded to :func:`write_pw_input`.
        base_kwargs: Extra keyword arguments forwarded to
            :func:`write_pw_input` for every point (e.g. ``{"ecutwfc": 80}``
            while sweeping kpts). The swept parameter must not appear here.

    Returns:
        Paths to the generated input files, in sweep order.

    Raises:
        ValueError: If ``param`` is not supported or appears in
            ``base_kwargs``.
    """
    if param not in SUPPORTED_SWEEP_PARAMETERS:
        raise ValueError(
            f"Unsupported sweep param={param!r}; "
            f"expected one of {sorted(SUPPORTED_SWEEP_PARAMETERS)}"
        )
    base = dict(base_kwargs or {})
    if param in base:
        raise ValueError(
            f"Sweep parameter {param!r} also appears in base_kwargs; "
            "remove it from base_kwargs to avoid conflicting values."
        )

    root = Path(out_root)
    written: list[Path] = []
    for value in values:
        label = _format_value(param, value)
        out_path = root / f"{param}_{label}" / "pw.in"
        kwargs: dict[str, Any] = dict(base)
        if param == "kpts":
            n = int(value)
            kwargs["kpts"] = (n, n, n)
        elif param == "hubbard_u":
            # Hubbard U is not a direct write_pw_input keyword; route it
            # through spin_and_hubbard_overrides. Spin polarization is
            # inferred from the structure's initial magnetic moments —
            # AFM CuO will be nspin=2 automatically. QE 7.1+ requires the
            # HUBBARD card (additional_cards=) instead of the deprecated
            # Hubbard_U(i) namelist keys.
            magnetic = any(
                m != 0.0 for m in atoms.get_initial_magnetic_moments()
            )
            nspin = 2 if magnetic else 1
            existing_extra = dict(kwargs.pop("extra_input_data", {}) or {})
            overrides = spin_and_hubbard_overrides(
                atoms, nspin=nspin, hubbard_u={"Cu": float(value)}
            )
            for namelist, entries in overrides.namelist_overrides.items():
                existing_extra.setdefault(namelist, {}).update(entries)
            kwargs["extra_input_data"] = existing_extra
            kwargs["additional_cards"] = overrides.hubbard_card
        else:
            kwargs[param] = float(value)
        write_pw_input(
            atoms,
            out_path=out_path,
            pseudopotentials=pseudopotentials,
            pseudo_dir=pseudo_dir,
            prefix=f"bulk_{param}_{label}",
            **kwargs,
        )
        written.append(out_path)
    return written
