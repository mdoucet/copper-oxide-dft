"""Tests for copper_oxide_dft.ml.fcp_rerank."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from copper_oxide_dft.config import SystemConfig
from copper_oxide_dft.ml.ensemble import phase_from_atoms
from copper_oxide_dft.ml.fcp_rerank import (
    DEFAULT_AG_AGCL_ABSOLUTE_POTENTIAL_V,
    DEFAULT_TARGET_POTENTIAL_V,
    FcpRerankResult,
    grand_potential_at_u,
    parse_fcp_tot_charge,
    prepare_fcp_inputs,
    rank_fcp_results,
    write_frontier_submit_scripts,
)
from copper_oxide_dft.structure_builder import (
    build_bulk_cu,
    build_bulk_cu2o,
    build_bulk_cuo,
)


def _assert_namelist_value(text: str, key: str, expected) -> None:
    """Anchored namelist-value check; see tests/test_ml_qe_driver.py for context."""
    pattern = rf"(?<![A-Za-z_]){re.escape(key)}\s*=\s*([-+0-9.eEdD]+)"
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    assert matches, f"Key {key!r} not found."
    found_values = [float(m.replace("D", "e").replace("d", "e")) for m in matches]
    assert any(np.isclose(v, float(expected)) for v in found_values), (
        f"Expected {key} = {expected}, found {found_values}"
    )


@pytest.fixture
def pseudo_dir(tmp_path: Path) -> Path:
    d = tmp_path / "pseudos"
    d.mkdir()
    (d / "Cu.upf").write_text("")
    (d / "O.upf").write_text("")
    return d


@pytest.fixture
def phase1_config() -> SystemConfig:
    return SystemConfig(
        ecutwfc_ry=100.0, kpts=(18, 18, 18), degauss_ry=0.01,
        extras={"lattice_a_ang": 3.6577},
    )


def _phase(atoms, energy=-100.0, mu_o=-6.5):
    return phase_from_atoms(atoms, energy_ev=energy, mu_o_ev=mu_o, source="unbiased")


# ---------- grand_potential_at_u ---------------------------------------------


def test_grand_potential_at_u_neutral_cell_equals_energy() -> None:
    """A run with tot_charge = 0 → Ω(U) = E_DFT exactly, regardless of U."""
    omega = grand_potential_at_u(
        energy_ev=-100.0, tot_charge=0.0, u_target_v=-0.8, reference_absolute_v=4.64,
    )
    assert omega == -100.0


def test_grand_potential_at_u_sign_convention() -> None:
    """At U = -0.8 V vs Ag/AgCl (μ_e = -3.84 eV), oxidising the cell (tot_charge > 0)
    must *lower* Ω (cell is at lower energy because we've removed expensive electrons).
    """
    e = -100.0
    omega_neutral = grand_potential_at_u(e, tot_charge=0.0, u_target_v=-0.8)
    omega_oxidised = grand_potential_at_u(e, tot_charge=+0.5, u_target_v=-0.8)
    omega_reduced = grand_potential_at_u(e, tot_charge=-0.5, u_target_v=-0.8)
    assert omega_oxidised < omega_neutral
    assert omega_reduced > omega_neutral


def test_grand_potential_at_u_arithmetic() -> None:
    """Ω(U) = E + μ_e · tot_charge with μ_e = -(V_abs + U).

    At U = -0.8, V_abs = 4.64: μ_e = -(4.64 - 0.8) = -3.84 eV.
    With tot_charge = 0.1: Ω = E - 3.84 · 0.1 = E - 0.384.
    """
    omega = grand_potential_at_u(
        energy_ev=-100.0,
        tot_charge=0.1,
        u_target_v=-0.8,
        reference_absolute_v=4.64,
    )
    np.testing.assert_allclose(omega, -100.384, atol=1e-6)


def test_default_potential_matches_pivot() -> None:
    assert DEFAULT_TARGET_POTENTIAL_V == -0.8
    assert DEFAULT_AG_AGCL_ABSOLUTE_POTENTIAL_V == 4.64


# ---------- parse_fcp_tot_charge ----------------------------------------------


def test_parse_tot_charge_finds_last_occurrence(tmp_path: Path) -> None:
    """If pw.out has multiple tot_charge lines (FCP iterates), pick the last."""
    out = tmp_path / "pw.out"
    out.write_text(
        "tot_charge =   0.500000\n"
        "tot_charge =   0.300000\n"
        "tot_charge =   0.281234\n"
    )
    assert parse_fcp_tot_charge(out) == pytest.approx(0.281234)


def test_parse_tot_charge_returns_none_when_absent(tmp_path: Path) -> None:
    out = tmp_path / "pw.out"
    out.write_text("This pw.out has no FCP section.\n")
    assert parse_fcp_tot_charge(out) is None


def test_parse_tot_charge_handles_fortran_d_exponent(tmp_path: Path) -> None:
    out = tmp_path / "pw.out"
    out.write_text("tot_charge =  1.234D-02\n")
    assert parse_fcp_tot_charge(out) == pytest.approx(1.234e-2)


def test_parse_tot_charge_handles_negative_values(tmp_path: Path) -> None:
    out = tmp_path / "pw.out"
    out.write_text("tot_charge =  -0.123\n")
    assert parse_fcp_tot_charge(out) == pytest.approx(-0.123)


def test_parse_tot_charge_does_not_match_prefixed_variants(tmp_path: Path) -> None:
    """Anchored regex must not match `new_tot_charge`, `my_tot_charge`, etc.

    The negative-look-behind lets us trust the "last occurrence" heuristic —
    otherwise a related debug print could shadow the real FCP-converged value.
    """
    out = tmp_path / "pw.out"
    out.write_text(
        "tot_charge =  0.100\n"          # real, should win
        "new_tot_charge =  0.999\n"      # prefixed variant, must NOT match
        "my_tot_charge =  -0.500\n"      # another prefixed variant
    )
    assert parse_fcp_tot_charge(out) == pytest.approx(0.100)


# ---------- prepare_fcp_inputs ------------------------------------------------


def test_prepare_fcp_inputs_creates_one_dir_per_candidate(
    tmp_path: Path, pseudo_dir: Path, phase1_config: SystemConfig
) -> None:
    candidates = [_phase(build_bulk_cu()), _phase(build_bulk_cu2o()), _phase(build_bulk_cuo())]
    paths = prepare_fcp_inputs(
        candidates, tmp_path / "rerank",
        system_config=phase1_config, pseudo_dir=pseudo_dir,
    )
    assert len(paths) == 3
    for i in range(3):
        assert (tmp_path / "rerank" / f"candidate_{i:02d}" / "pw.in").is_file()


def test_prepare_fcp_inputs_writes_fcp_keywords(
    tmp_path: Path, pseudo_dir: Path, phase1_config: SystemConfig
) -> None:
    candidates = [_phase(build_bulk_cu2o())]
    prepare_fcp_inputs(
        candidates, tmp_path / "rerank",
        system_config=phase1_config, pseudo_dir=pseudo_dir,
    )
    text = (tmp_path / "rerank" / "candidate_00" / "pw.in").read_text().lower()
    assert "lfcp" in text
    assert "assume_isolated" in text and "esm" in text
    assert "fcp_mu" in text


def test_prepare_fcp_inputs_writes_spin_and_hubbard_for_cu_o(
    tmp_path: Path, pseudo_dir: Path, phase1_config: SystemConfig
) -> None:
    candidates = [_phase(build_bulk_cu2o())]
    prepare_fcp_inputs(
        candidates, tmp_path / "rerank",
        system_config=phase1_config, pseudo_dir=pseudo_dir,
    )
    text = (tmp_path / "rerank" / "candidate_00" / "pw.in").read_text()
    assert "nspin" in text.lower()
    # QE 7.1+ HUBBARD card; the old &SYSTEM Hubbard_U(i) keys would
    # cause QE to abort with "DFT+Hubbard input syntax has changed".
    assert "HUBBARD {atomic}" in text
    assert "U Cu-3d" in text
    assert "hubbard_u(1)" not in text.lower()


def test_prepare_fcp_inputs_rejects_empty_candidates(
    tmp_path: Path, pseudo_dir: Path, phase1_config: SystemConfig
) -> None:
    with pytest.raises(ValueError):
        prepare_fcp_inputs(
            [], tmp_path / "rerank",
            system_config=phase1_config, pseudo_dir=pseudo_dir,
        )


def test_prepare_fcp_inputs_uses_phase1_ecutwfc(
    tmp_path: Path, pseudo_dir: Path, phase1_config: SystemConfig
) -> None:
    candidates = [_phase(build_bulk_cu())]
    prepare_fcp_inputs(
        candidates, tmp_path / "rerank",
        system_config=phase1_config, pseudo_dir=pseudo_dir,
    )
    text = (tmp_path / "rerank" / "candidate_00" / "pw.in").read_text()
    _assert_namelist_value(text, "ecutwfc", 100.0)


# ---------- write_frontier_submit_scripts ------------------------------------


def test_write_frontier_submit_scripts_one_per_pwin(
    tmp_path: Path, pseudo_dir: Path, phase1_config: SystemConfig
) -> None:
    candidates = [_phase(build_bulk_cu()), _phase(build_bulk_cu2o())]
    prepare_fcp_inputs(
        candidates, tmp_path / "rerank",
        system_config=phase1_config, pseudo_dir=pseudo_dir,
    )
    scripts = write_frontier_submit_scripts(tmp_path / "rerank", account="DUMMY")
    assert len(scripts) == 2
    for s in scripts:
        assert s.name == "submit.sh"
        assert s.is_file()


def test_write_frontier_submit_scripts_use_frontier_conventions(
    tmp_path: Path, pseudo_dir: Path, phase1_config: SystemConfig
) -> None:
    candidates = [_phase(build_bulk_cu())]
    prepare_fcp_inputs(
        candidates, tmp_path / "rerank",
        system_config=phase1_config, pseudo_dir=pseudo_dir,
    )
    scripts = write_frontier_submit_scripts(tmp_path / "rerank", account="ABC123")
    text = scripts[0].read_text()
    assert "ABC123" in text
    # Frontier preset: 8 GCDs/node, 7 cores/rank.
    assert "--gpus-per-node=8" in text
    assert "-c 7" in text
    assert "MPICH_GPU_SUPPORT_ENABLED" in text


# ---------- rank_fcp_results --------------------------------------------------


def _write_fake_fcp_pw_out(
    path: Path, *, energy_ry: float, tot_charge: float, job_done: bool = True
) -> None:
    text = (
        f"!    total energy              =  {energy_ry:.6f} Ry\n"
        f"tot_charge =  {tot_charge:.6f}\n"
        + ("JOB DONE.\n" if job_done else "")
    )
    path.write_text(text)


def test_rank_fcp_results_returns_empty_for_empty_dir(tmp_path: Path) -> None:
    assert rank_fcp_results(tmp_path) == []


def test_rank_fcp_results_orders_by_omega_u(tmp_path: Path) -> None:
    out_root = tmp_path / "rerank"
    out_root.mkdir()
    # Three candidates: low energy + small charge, mid energy + large charge,
    # high energy + neg charge.
    for i, (e_ry, q) in enumerate([(-100.0, 0.0), (-99.0, 0.2), (-98.0, -0.3)]):
        d = out_root / f"candidate_{i:02d}"
        d.mkdir()
        _write_fake_fcp_pw_out(d / "pw.out", energy_ry=e_ry, tot_charge=q)

    results = rank_fcp_results(out_root, u_target_v=-0.8, reference_absolute_v=4.64)
    assert len(results) == 3
    # Sort key: Ω(U); the most-oxidised candidate should win at U = -0.8 V.
    omegas = [r.omega_u_ev for r in results]
    assert omegas == sorted(omegas)


def test_rank_fcp_results_unconverged_trail(tmp_path: Path) -> None:
    """Candidates without tot_charge appear at the end with omega_u_ev = None."""
    out_root = tmp_path / "rerank"
    out_root.mkdir()

    # Converged candidate.
    d0 = out_root / "candidate_00"
    d0.mkdir()
    _write_fake_fcp_pw_out(d0 / "pw.out", energy_ry=-100.0, tot_charge=0.1)

    # Unconverged: pw.out has energy but no FCP block.
    d1 = out_root / "candidate_01"
    d1.mkdir()
    (d1 / "pw.out").write_text(
        "!    total energy              =  -99.000000 Ry\n"
        "JOB DONE.\n"
    )

    results = rank_fcp_results(out_root)
    assert len(results) == 2
    # First should be converged, second unconverged.
    assert results[0].omega_u_ev is not None
    assert results[1].omega_u_ev is None
    assert results[1].tot_charge is None


def test_rank_fcp_results_skips_missing_pw_out(tmp_path: Path) -> None:
    out_root = tmp_path / "rerank"
    (out_root / "candidate_00").mkdir(parents=True)
    # No pw.out yet — the run hasn't happened.
    assert rank_fcp_results(out_root) == []


def test_rank_fcp_results_handles_completely_failed_run(tmp_path: Path) -> None:
    """pw.out exists but has no convergence and no FCP section.
    Should produce a result with NaN energy and None omega."""
    out_root = tmp_path / "rerank"
    d = out_root / "candidate_00"
    d.mkdir(parents=True)
    (d / "pw.out").write_text("Error: run aborted.\n")
    results = rank_fcp_results(out_root)
    assert len(results) == 1
    assert results[0].omega_u_ev is None
    assert results[0].tot_charge is None


# ---------- FcpRerankResult dataclass ----------------------------------------


def test_fcp_rerank_result_dataclass_carries_target_potential() -> None:
    r = FcpRerankResult(
        candidate_id="candidate_07",
        atoms=None,
        energy_ev=-100.0,
        tot_charge=0.123,
        omega_u_ev=-100.473,
        u_target_v=-0.8,
    )
    assert r.candidate_id == "candidate_07"
    assert r.u_target_v == -0.8
