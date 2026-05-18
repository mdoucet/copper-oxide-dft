"""Convergence analysis: turn a sweep tree into a converged-parameter pick.

Given a directory produced by :func:`copper_oxide_dft.convergence.sweep_convergence`
(or the ``sweep`` CLI), this module:

1. Walks the tree, parses each ``pw.out``, pulls the total energy.
2. Reports energy-vs-parameter for the swept knob (``ecutwfc`` / ``kpts``
   / ``degauss`` / ``hubbard_u``).
3. Picks the smallest parameter value that converges total energy per
   atom to within a user-chosen threshold (default 1 meV/atom — the
   Phase 1 success criterion in [implementation-plan.md](../../docs/implementation-plan.md)).
4. Optionally renders a matplotlib convergence plot for the lab notebook.

The point estimator is per-atom so the same threshold works across
systems of different size.

**Asymptote direction is parameter-dependent**:

* ``ecutwfc``, ``kpts``, ``hubbard_u``: larger = better. The largest
  sweep value is the asymptote; the smallest value within threshold is
  the converged pick.
* ``degauss``: smaller = better (the T→0 physical limit; large smearing
  *distorts* the Fermi surface rather than converging). The smallest
  sweep value is the asymptote; the largest value within threshold is
  the converged pick.

The set :data:`LOW_VALUE_IS_ASYMPTOTE_PARAMS` tracks which parameters
treat the smallest value as the asymptote. :func:`analyze_sweep` picks
the direction automatically; callers of :func:`find_converged_value`
can override with the ``low_value_is_asymptote`` keyword.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from copper_oxide_dft.parse import parse_pw_output

if TYPE_CHECKING:
    from matplotlib.axes import Axes

DEFAULT_CONVERGENCE_THRESHOLD_MEV_PER_ATOM = 1.0
"""Default per-atom energy-difference threshold for declaring convergence.
Matches the Phase 1 success criterion."""

LOW_VALUE_IS_ASYMPTOTE_PARAMS = frozenset({"degauss"})
"""Sweep parameters whose *smallest* value is the converged asymptote.

``degauss`` is the T→0 limit at zero smearing; values larger than the
plateau wash out the Fermi surface and give wrong energies, not better
ones. All other supported parameters (``ecutwfc``, ``kpts``,
``hubbard_u``) are the conventional "larger = better" direction."""


@dataclass(frozen=True)
class SweepPoint:
    """One point on a convergence curve: parameter value + total energy."""

    param_value: float
    total_energy_ev: float
    n_atoms: int
    job_done: bool
    source_path: Path

    @property
    def energy_per_atom_ev(self) -> float:
        if self.n_atoms <= 0:
            raise ValueError(f"n_atoms must be positive; got {self.n_atoms}")
        return self.total_energy_ev / self.n_atoms


@dataclass(frozen=True)
class ConvergenceResult:
    """Outcome of analyzing a sweep tree.

    Attributes:
        param_name: The swept parameter ('ecutwfc', 'kpts', 'degauss',
            'hubbard_u').
        points: Sweep points in ascending parameter order.
        converged_value: Parameter value at which the per-atom energy
            is within ``threshold_mev_per_atom`` of the asymptote.
            ``None`` if no value converges. For ``degauss`` this is the
            *largest* value within threshold (the converged pick walks
            up from the T→0 asymptote); for every other parameter it
            is the *smallest* value within threshold (the converged pick
            walks down from the high-resolution asymptote).
        threshold_mev_per_atom: The threshold used.
        low_value_is_asymptote: If True, the smallest sweep value was
            treated as the asymptote (degauss convention). If False,
            the largest sweep value was treated as the asymptote
            (ecutwfc / kpts / hubbard_u convention).
    """

    param_name: str
    points: tuple[SweepPoint, ...]
    converged_value: float | None
    threshold_mev_per_atom: float
    low_value_is_asymptote: bool = False


_SWEEP_DIR_RE = re.compile(r"^(?P<param>[a-z_]+)_(?P<value>[-+0-9p.]+)$")


def collect_sweep_points(
    root: str | Path, *, input_file_name: str = "pw.in"
) -> tuple[str, list[SweepPoint]]:
    """Walk a sweep tree and parse each ``pw.out``.

    The directory naming convention from :mod:`copper_oxide_dft.convergence`
    is ``<root>/<param>_<value>/{pw.in, pw.out}``; this function discovers
    those, parses each output, and reports the (param_value, energy, atoms)
    triples in ascending parameter order.

    Args:
        root: Root of the sweep tree.
        input_file_name: Name of the QE input file; used to count atoms in
            each sweep point. The matching output is the same stem with
            ``.out`` extension.

    Returns:
        Tuple of ``(param_name, points)`` where ``param_name`` is shared
        across all subdirectories under ``root``.

    Raises:
        ValueError: If the tree mixes multiple swept parameters or
            contains zero recognized subdirectories.
        FileNotFoundError: If a subdirectory's ``pw.out`` is missing.
    """
    root_path = Path(root)
    subdirs = sorted(d for d in root_path.iterdir() if d.is_dir())

    discovered: list[tuple[str, float, Path]] = []
    for sub in subdirs:
        match = _SWEEP_DIR_RE.match(sub.name)
        if not match:
            continue
        param = match.group("param")
        # Restore the "0p3" → 0.3 transform that convergence.py uses for
        # degauss labels; integer params come through unchanged.
        raw_value = match.group("value").replace("p", ".")
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"Could not parse parameter value from {sub.name!r}"
            ) from exc
        discovered.append((param, value, sub))

    if not discovered:
        raise ValueError(
            f"No sweep subdirectories found under {root_path}. "
            "Expected '<param>_<value>' directory names."
        )
    param_names = {p for p, _, _ in discovered}
    if len(param_names) > 1:
        raise ValueError(
            f"Sweep tree mixes parameters {sorted(param_names)}; "
            "each tree must sweep one parameter at a time."
        )
    (param_name,) = param_names

    points: list[SweepPoint] = []
    for _, value, subdir in sorted(discovered, key=lambda t: t[1]):
        out_file = subdir / "pw.out"
        if not out_file.is_file():
            raise FileNotFoundError(f"Missing pw.out in {subdir}")
        result = parse_pw_output(out_file)
        n_atoms = _count_atoms(subdir / input_file_name)
        points.append(
            SweepPoint(
                param_value=value,
                total_energy_ev=float(result.total_energy_ev),
                n_atoms=n_atoms,
                job_done=result.job_done,
                source_path=out_file,
            )
        )
    return param_name, points


def find_converged_value(
    points: Iterable[SweepPoint],
    *,
    threshold_mev_per_atom: float = DEFAULT_CONVERGENCE_THRESHOLD_MEV_PER_ATOM,
    low_value_is_asymptote: bool = False,
) -> float | None:
    """Cheapest sweep value within ``threshold`` of the converged asymptote.

    The asymptote is one end of the sweep curve; the converged pick is
    the value on the other end that first lies within the threshold of
    the asymptote. The asymptote itself is excluded (a single point
    can't prove its own convergence).

    For ``ecutwfc`` / ``kpts`` / ``hubbard_u`` (``low_value_is_asymptote=False``):
    the asymptote is the *largest* sweep value, and the converged pick
    is the *smallest* value within threshold (cheapest setting that's
    good enough).

    For ``degauss`` (``low_value_is_asymptote=True``): the asymptote is
    the *smallest* sweep value (T→0 limit), and the converged pick is
    the *largest* value within threshold (loosest smearing that still
    matches T→0 — useful because tighter smearing demands denser k-points).

    Args:
        points: Sweep points (need not be pre-sorted).
        threshold_mev_per_atom: Convergence threshold (meV/atom).
        low_value_is_asymptote: If True, treat the smallest value as the
            asymptote. See :data:`LOW_VALUE_IS_ASYMPTOTE_PARAMS`.

    Returns:
        Converged parameter value, or ``None`` if no point qualifies.
    """
    ordered = sorted(points, key=lambda p: p.param_value)
    if len(ordered) < 2:
        return None
    threshold_ev = threshold_mev_per_atom * 1.0e-3
    if low_value_is_asymptote:
        asymptote = ordered[0].energy_per_atom_ev
        # Walk from largest toward smallest; first hit is the largest
        # value within threshold (loosest setting still matching T→0).
        for point in reversed(ordered[1:]):
            if abs(point.energy_per_atom_ev - asymptote) <= threshold_ev:
                return point.param_value
        return None
    asymptote = ordered[-1].energy_per_atom_ev
    for point in ordered[:-1]:
        if abs(point.energy_per_atom_ev - asymptote) <= threshold_ev:
            return point.param_value
    return None


def analyze_sweep(
    root: str | Path,
    *,
    threshold_mev_per_atom: float = DEFAULT_CONVERGENCE_THRESHOLD_MEV_PER_ATOM,
) -> ConvergenceResult:
    """End-to-end: walk a sweep tree, parse outputs, pick converged value.

    The asymptote direction is chosen automatically based on the swept
    parameter (see :data:`LOW_VALUE_IS_ASYMPTOTE_PARAMS`).
    """
    param_name, points = collect_sweep_points(root)
    low_is_asymptote = param_name in LOW_VALUE_IS_ASYMPTOTE_PARAMS
    converged = find_converged_value(
        points,
        threshold_mev_per_atom=threshold_mev_per_atom,
        low_value_is_asymptote=low_is_asymptote,
    )
    return ConvergenceResult(
        param_name=param_name,
        points=tuple(points),
        converged_value=converged,
        threshold_mev_per_atom=threshold_mev_per_atom,
        low_value_is_asymptote=low_is_asymptote,
    )


def plot_convergence(
    result: ConvergenceResult,
    *,
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """Plot per-atom energy vs. swept parameter with the threshold band.

    The asymptote (largest sweep value) gets a horizontal reference line;
    the threshold band is shaded ±``threshold_mev_per_atom`` around it.
    A vertical line marks the converged value.

    Args:
        result: Output of :func:`analyze_sweep`.
        ax: Existing Axes to draw into; new figure+axes is created if None.
        title: Custom plot title.

    Returns:
        The matplotlib Axes.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    xs = [p.param_value for p in result.points]
    ys = [p.energy_per_atom_ev for p in result.points]
    ax.plot(xs, ys, marker="o", color="C0", label="E/atom")

    asymptote = ys[0] if result.low_value_is_asymptote else ys[-1]
    threshold_ev = result.threshold_mev_per_atom * 1.0e-3
    ax.axhline(asymptote, color="gray", linestyle="--", linewidth=0.8)
    ax.axhspan(asymptote - threshold_ev, asymptote + threshold_ev, color="gray", alpha=0.15)

    if result.converged_value is not None:
        ax.axvline(
            result.converged_value,
            color="C2",
            linestyle=":",
            label=f"converged at {result.param_name}={result.converged_value:g}",
        )

    ax.set_xlabel(result.param_name)
    ax.set_ylabel("E / atom (eV)")
    if title is None:
        title = (
            f"Convergence vs. {result.param_name} "
            f"(threshold {result.threshold_mev_per_atom} meV/atom)"
        )
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    return ax


def _count_atoms(pw_in_path: Path) -> int:
    """Read a pw.x input file and return the atom count from ATOMIC_POSITIONS.

    Used to normalize total energies per atom. Reading the input rather
    than the output keeps this robust to QE versions: the input layout
    we write ourselves and is stable.
    """
    from ase.io.espresso import read_espresso_in

    with pw_in_path.open() as fh:
        atoms = read_espresso_in(fh)
    return len(atoms)


__all__ = (
    "DEFAULT_CONVERGENCE_THRESHOLD_MEV_PER_ATOM",
    "ConvergenceResult",
    "SweepPoint",
    "analyze_sweep",
    "collect_sweep_points",
    "find_converged_value",
    "plot_convergence",
)
