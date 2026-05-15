"""Tests for copper_oxide_dft.analysis (convergence-sweep analyzer)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")  # noqa: E402 — must precede pyplot import

from copper_oxide_dft.analysis import (
    SweepPoint,
    analyze_sweep,
    collect_sweep_points,
    find_converged_value,
    plot_convergence,
)
from copper_oxide_dft.qe_input import write_pw_input
from copper_oxide_dft.structure_builder import build_bulk_cu


@pytest.fixture
def pseudo_dir(tmp_path: Path) -> Path:
    d = tmp_path / "pseudos"
    d.mkdir()
    (d / "Cu.upf").write_text("")
    return d


def _make_sweep_dir(
    root: Path, param: str, label: str, energy_ry: float, pseudo_dir: Path
) -> Path:
    """Build a sweep subdirectory with a real pw.in and a synthetic pw.out."""
    sub = root / f"{param}_{label}"
    sub.mkdir(parents=True)
    write_pw_input(
        build_bulk_cu(),
        out_path=sub / "pw.in",
        pseudopotentials={"Cu": "Cu.upf"},
        pseudo_dir=pseudo_dir,
    )
    (sub / "pw.out").write_text(
        f"!    total energy = {energy_ry} Ry\n"
        "the Fermi energy is 3.0 ev\n"
        "JOB DONE.\n"
    )
    return sub


# ---- find_converged_value --------------------------------------------------


def _sp(value: float, energy_per_atom_ev: float) -> SweepPoint:
    return SweepPoint(
        param_value=value,
        total_energy_ev=energy_per_atom_ev,
        n_atoms=1,
        job_done=True,
        source_path=Path("/tmp/fake"),
    )


def test_find_converged_value_returns_smallest_within_threshold() -> None:
    # Asymptote = -100 eV. Points within 1 meV of that are 'converged'.
    points = [
        _sp(40.0, -99.990),  # 10 meV off — not converged
        _sp(60.0, -99.9995),  # 0.5 meV off — converged
        _sp(80.0, -100.0001),  # 0.1 meV — converged
        _sp(100.0, -100.0000),  # asymptote
    ]
    assert find_converged_value(points, threshold_mev_per_atom=1.0) == 60.0


def test_find_converged_value_returns_none_when_curve_not_plateaued() -> None:
    points = [_sp(40.0, -99.0), _sp(60.0, -99.5), _sp(80.0, -99.8)]
    # Asymptote is -99.8 eV/atom, 200 meV from the first point; none within 1 meV.
    assert find_converged_value(points, threshold_mev_per_atom=1.0) is None


def test_find_converged_value_single_point_returns_none() -> None:
    assert find_converged_value([_sp(40.0, -100.0)]) is None


# ---- collect_sweep_points + analyze_sweep ----------------------------------


def test_collect_sweep_points_parses_tree(tmp_path: Path, pseudo_dir: Path) -> None:
    root = tmp_path / "conv"
    _make_sweep_dir(root, "ecutwfc", "40", -100.0, pseudo_dir)
    _make_sweep_dir(root, "ecutwfc", "60", -100.5, pseudo_dir)
    _make_sweep_dir(root, "ecutwfc", "80", -100.6, pseudo_dir)

    param, points = collect_sweep_points(root)
    assert param == "ecutwfc"
    assert [p.param_value for p in points] == [40.0, 60.0, 80.0]
    # 1 Ry ≈ 13.605693 eV; sanity-check the parsed conversion.
    assert points[0].total_energy_ev == pytest.approx(-100.0 * 13.605693, rel=1e-5)
    assert all(p.job_done for p in points)


def test_collect_sweep_points_handles_p_in_degauss_labels(
    tmp_path: Path, pseudo_dir: Path
) -> None:
    """convergence.py writes degauss values as '0p020'; the analyzer must decode that."""
    root = tmp_path / "conv"
    _make_sweep_dir(root, "degauss", "0p010", -100.0, pseudo_dir)
    _make_sweep_dir(root, "degauss", "0p020", -100.0001, pseudo_dir)
    param, points = collect_sweep_points(root)
    assert param == "degauss"
    assert points[0].param_value == 0.010
    assert points[1].param_value == 0.020


def test_collect_sweep_points_rejects_mixed_parameters(
    tmp_path: Path, pseudo_dir: Path
) -> None:
    root = tmp_path / "conv"
    _make_sweep_dir(root, "ecutwfc", "40", -100.0, pseudo_dir)
    _make_sweep_dir(root, "kpts", "8", -100.0, pseudo_dir)
    with pytest.raises(ValueError, match="mixes parameters"):
        collect_sweep_points(root)


def test_collect_sweep_points_rejects_empty_tree(tmp_path: Path) -> None:
    (tmp_path / "conv").mkdir()
    with pytest.raises(ValueError, match="No sweep subdirectories"):
        collect_sweep_points(tmp_path / "conv")


def test_analyze_sweep_end_to_end(tmp_path: Path, pseudo_dir: Path) -> None:
    """Tight curve: ecutwfc=60 already converged within 1 meV/atom (per-atom
    asymptote = -1360.5694 eV; offsets ~ 1 Ry = 13.6 eV per row to stay
    well clear at the unconverged end)."""
    root = tmp_path / "conv"
    _make_sweep_dir(root, "ecutwfc", "40", -100.0, pseudo_dir)
    _make_sweep_dir(root, "ecutwfc", "60", -100.06, pseudo_dir)
    _make_sweep_dir(root, "ecutwfc", "80", -100.0600001, pseudo_dir)

    result = analyze_sweep(root, threshold_mev_per_atom=10.0)
    assert result.param_name == "ecutwfc"
    assert len(result.points) == 3
    assert result.converged_value == 60.0


# ---- plot_convergence ------------------------------------------------------


def test_plot_convergence_axes_labels(tmp_path: Path, pseudo_dir: Path) -> None:
    import matplotlib.pyplot as plt

    root = tmp_path / "conv"
    _make_sweep_dir(root, "ecutwfc", "40", -100.0, pseudo_dir)
    _make_sweep_dir(root, "ecutwfc", "60", -100.06, pseudo_dir)
    _make_sweep_dir(root, "ecutwfc", "80", -100.0600001, pseudo_dir)

    result = analyze_sweep(root, threshold_mev_per_atom=10.0)
    fig, ax = plt.subplots()
    ax = plot_convergence(result, ax=ax)
    assert ax.get_xlabel() == "ecutwfc"
    assert "E / atom" in ax.get_ylabel()
    plt.close(fig)
