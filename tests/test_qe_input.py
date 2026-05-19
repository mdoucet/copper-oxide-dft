"""Tests for copper_oxide_dft.qe_input."""

from __future__ import annotations

from pathlib import Path

import pytest

from copper_oxide_dft.qe_input import (
    DEFAULT_PSEUDOPOTENTIALS,
    EV_PER_RYDBERG,
    PSEUDO_DIR_ENV_VAR,
    SHE_ABSOLUTE_POTENTIAL_V,
    SUPPORTED_CALCULATIONS,
    fcp_overrides_for_potential,
    merge_namelist_overrides,
    spin_and_hubbard_overrides,
    write_hp_input,
    write_pw_input,
)
from copper_oxide_dft.structure_builder import (
    build_bulk_cu,
    build_bulk_cu2o,
    build_bulk_cuo,
)


@pytest.fixture
def pseudo_dir(tmp_path: Path) -> Path:
    d = tmp_path / "pseudos"
    d.mkdir()
    (d / "Cu.upf").write_text("")
    return d


def test_write_pw_input_explicit_pseudo_dir(tmp_path: Path, pseudo_dir: Path) -> None:
    atoms = build_bulk_cu()
    out_path = tmp_path / "run" / "bulk_cu.in"

    written = write_pw_input(
        atoms,
        out_path=out_path,
        pseudopotentials={"Cu": "Cu.upf"},
        ecutwfc=60.0,
        kpts=(6, 6, 6),
        pseudo_dir=pseudo_dir,
    )

    assert written == out_path
    text = out_path.read_text()
    # Namelists
    assert "&CONTROL" in text
    assert "&SYSTEM" in text
    assert "&ELECTRONS" in text
    # Project-standard parameters
    assert "calculation" in text and "scf" in text
    assert "ecutwfc" in text and "60" in text
    assert "smearing" in text and "mv" in text.lower()  # Marzari-Vanderbilt
    assert "K_POINTS" in text
    assert "Cu.upf" in text


def test_write_pw_input_falls_back_to_env_var(
    tmp_path: Path, pseudo_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(PSEUDO_DIR_ENV_VAR, str(pseudo_dir))
    atoms = build_bulk_cu()
    out_path = tmp_path / "bulk_cu.in"

    write_pw_input(
        atoms,
        out_path=out_path,
        pseudopotentials={"Cu": "Cu.upf"},
    )

    assert out_path.is_file()


def test_write_pw_input_raises_when_no_pseudo_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(PSEUDO_DIR_ENV_VAR, raising=False)
    atoms = build_bulk_cu()
    with pytest.raises(FileNotFoundError, match=PSEUDO_DIR_ENV_VAR):
        write_pw_input(
            atoms,
            out_path=tmp_path / "x.in",
            pseudopotentials={"Cu": "Cu.upf"},
        )


def test_write_pw_input_raises_when_pseudo_dir_missing(tmp_path: Path) -> None:
    atoms = build_bulk_cu()
    with pytest.raises(FileNotFoundError, match="does not exist"):
        write_pw_input(
            atoms,
            out_path=tmp_path / "x.in",
            pseudopotentials={"Cu": "Cu.upf"},
            pseudo_dir=tmp_path / "nonexistent",
        )


def test_write_pw_input_extra_namelist_override(
    tmp_path: Path, pseudo_dir: Path
) -> None:
    """Spin-polarized override is the use case we'll need for CuO later."""
    atoms = build_bulk_cu()
    out_path = tmp_path / "bulk_cu.in"
    write_pw_input(
        atoms,
        out_path=out_path,
        pseudopotentials={"Cu": "Cu.upf"},
        pseudo_dir=pseudo_dir,
        extra_input_data={"system": {"nspin": 2}},
    )
    assert "nspin" in out_path.read_text()


def test_write_pw_input_vc_relax_adds_ions_and_cell_namelists(
    tmp_path: Path, pseudo_dir: Path
) -> None:
    """vc-relax must emit IONS + CELL namelists with BFGS dynamics; this is
    what we'll use for Phase 1 bulk-Cu lattice optimization."""
    atoms = build_bulk_cu()
    out_path = tmp_path / "vc_relax.in"
    write_pw_input(
        atoms,
        out_path=out_path,
        pseudopotentials={"Cu": "Cu.upf"},
        pseudo_dir=pseudo_dir,
        calculation="vc-relax",
    )
    text = out_path.read_text()
    assert "'vc-relax'" in text
    assert "ion_dynamics" in text
    assert "cell_dynamics" in text
    assert "bfgs" in text


def test_write_pw_input_relax_adds_ions_but_not_cell(
    tmp_path: Path, pseudo_dir: Path
) -> None:
    atoms = build_bulk_cu()
    out_path = tmp_path / "relax.in"
    write_pw_input(
        atoms,
        out_path=out_path,
        pseudopotentials={"Cu": "Cu.upf"},
        pseudo_dir=pseudo_dir,
        calculation="relax",
    )
    text = out_path.read_text()
    assert "'relax'" in text
    assert "ion_dynamics" in text
    assert "cell_dynamics" not in text


def test_write_pw_input_rejects_bad_calculation(
    tmp_path: Path, pseudo_dir: Path
) -> None:
    atoms = build_bulk_cu()
    with pytest.raises(ValueError, match="Unsupported calculation"):
        write_pw_input(
            atoms,
            out_path=tmp_path / "x.in",
            pseudopotentials={"Cu": "Cu.upf"},
            pseudo_dir=pseudo_dir,
            calculation="totally-not-a-thing",
        )


def test_supported_calculations_includes_expected_modes() -> None:
    assert {"scf", "relax", "vc-relax"}.issubset(SUPPORTED_CALCULATIONS)


# ---- spin_and_hubbard_overrides -------------------------------------------


def test_spin_and_hubbard_overrides_emits_hubbard_card_for_cu2o() -> None:
    """Cu2O: Cu species labelled 'Cu' in the new HUBBARD card."""
    cu2o = build_bulk_cu2o()
    overrides = spin_and_hubbard_overrides(cu2o, nspin=1, hubbard_u={"Cu": 4.0})
    # No Hubbard_U(i) in &SYSTEM — QE 7.1+ rejects that syntax.
    system = overrides.namelist_overrides["system"]
    assert system["nspin"] == 1
    assert not any(k.startswith("Hubbard_U(") for k in system)
    # The card carries the U on Cu-3d.
    assert "HUBBARD { ortho-atomic }" in overrides.hubbard_card
    assert "U Cu-3d 4.000000" in overrides.hubbard_card


def test_spin_and_hubbard_overrides_no_card_when_no_hubbard_u() -> None:
    """Plain spin run (no Hubbard U) → empty card, namelist still set."""
    cuo = build_bulk_cuo()
    overrides = spin_and_hubbard_overrides(cuo, nspin=2)
    assert overrides.namelist_overrides["system"]["nspin"] == 2
    assert overrides.hubbard_card == ""


def test_spin_and_hubbard_overrides_silently_skips_absent_species() -> None:
    """Bulk Cu has no O; an O entry in hubbard_u should be ignored cleanly."""
    cu = build_bulk_cu()
    overrides = spin_and_hubbard_overrides(
        cu, nspin=1, hubbard_u={"Cu": 4.0, "O": 99.0}
    )
    # Only Cu shows up in the card (O is absent from the structure).
    assert "U Cu-3d 4.000000" in overrides.hubbard_card
    assert "O-2p" not in overrides.hubbard_card


def test_spin_and_hubbard_overrides_rejects_invalid_nspin() -> None:
    cu = build_bulk_cu()
    with pytest.raises(ValueError, match="nspin"):
        spin_and_hubbard_overrides(cu, nspin=3)


def test_spin_and_hubbard_overrides_starting_magnetization_as_mapping() -> None:
    """Symbol mapping replicates the value across AFM-split sub-species."""
    cuo = build_bulk_cuo()
    overrides = spin_and_hubbard_overrides(
        cuo, nspin=2, starting_magnetization={"Cu": 1.0, "O": 0.0}
    )
    system = overrides.namelist_overrides["system"]
    # CuO has 3 species under AFM splitting: (Cu,+1), (Cu,-1), (O,0).
    assert system["starting_magnetization(1)"] == 1.0  # Cu+
    assert system["starting_magnetization(2)"] == 1.0  # Cu- (still chemically Cu)
    assert system["starting_magnetization(3)"] == 0.0  # O


def test_spin_and_hubbard_overrides_starting_magnetization_as_sequence() -> None:
    """Sequence form indexes by species position, not by chemical symbol."""
    cuo = build_bulk_cuo()
    overrides = spin_and_hubbard_overrides(
        cuo, nspin=2, starting_magnetization=[0.7, -0.7, 0.0]
    )
    system = overrides.namelist_overrides["system"]
    assert system["starting_magnetization(1)"] == 0.7
    assert system["starting_magnetization(2)"] == -0.7
    assert system["starting_magnetization(3)"] == 0.0


def test_spin_and_hubbard_overrides_compose_with_write_pw_input(
    tmp_path: Path, pseudo_dir: Path
) -> None:
    """End-to-end: namelist + HUBBARD card both land in the file.

    ASE writes namelist keys in lowercase regardless of input case, so the
    test is case-insensitive for those; the HUBBARD card preserves case.
    """
    (pseudo_dir / "O.upf").write_text("")
    cu2o = build_bulk_cu2o()
    out_path = tmp_path / "cu2o.in"
    spin = spin_and_hubbard_overrides(cu2o, nspin=1, hubbard_u={"Cu": 4.0})
    write_pw_input(
        cu2o,
        out_path=out_path,
        pseudopotentials={"Cu": "Cu.upf", "O": "O.upf"},
        pseudo_dir=pseudo_dir,
        calculation="vc-relax",
        extra_input_data=spin.namelist_overrides,
        additional_cards=spin.hubbard_card,
    )
    text = out_path.read_text()
    # Namelist piece (case-insensitive).
    assert "nspin" in text.lower()
    # Card piece (case-preserving).
    assert "HUBBARD { ortho-atomic }" in text
    assert "U Cu-3d 4.000000" in text
    # And the deprecated form must NOT appear — that's exactly what QE 7.1+ rejects.
    assert "hubbard_u(1)" not in text.lower()


def test_spin_and_hubbard_overrides_afm_cuo_splits_cu_into_two_species(
    tmp_path: Path, pseudo_dir: Path
) -> None:
    """The bug we hit on first run: AFM CuO has TWO Cu species and BOTH need U."""
    (pseudo_dir / "O.upf").write_text("")
    cuo = build_bulk_cuo()
    overrides = spin_and_hubbard_overrides(cuo, nspin=2, hubbard_u={"Cu": 4.0})
    # Both Cu sublattices appear in the card under their ASE labels (Cu, Cu1).
    assert "U Cu-3d 4.000000" in overrides.hubbard_card
    assert "U Cu1-3d 4.000000" in overrides.hubbard_card

    # The QE input file also reflects the doubled entry.
    out_path = tmp_path / "cuo.in"
    write_pw_input(
        cuo,
        out_path=out_path,
        pseudopotentials={"Cu": "Cu.upf", "O": "O.upf"},
        pseudo_dir=pseudo_dir,
        calculation="vc-relax",
        extra_input_data=overrides.namelist_overrides,
        additional_cards=overrides.hubbard_card,
    )
    text = out_path.read_text()
    assert "U Cu-3d 4.000000" in text
    assert "U Cu1-3d 4.000000" in text


def test_spin_and_hubbard_overrides_unknown_manifold_raises() -> None:
    """An unregistered species in hubbard_u must raise (not silently drop U)."""
    from ase import Atoms

    # Krypton has no entry in DEFAULT_HUBBARD_MANIFOLDS — synthetic example.
    kr = Atoms("Kr", positions=[(0, 0, 0)], cell=[5, 5, 5], pbc=True)
    with pytest.raises(KeyError, match="Hubbard manifold"):
        spin_and_hubbard_overrides(kr, nspin=1, hubbard_u={"Kr": 4.0})


def test_spin_and_hubbard_overrides_projector_type_propagates() -> None:
    """Non-default projector flows through to the card header.

    Uses ``atomic`` (the legacy projector — known to fail for AFM CuO
    with our PP, but a useful sanity-check value to confirm the
    ``projector_type=`` kwarg is honoured).
    """
    cu2o = build_bulk_cu2o()
    overrides = spin_and_hubbard_overrides(
        cu2o, nspin=1, hubbard_u={"Cu": 4.0}, projector_type="atomic"
    )
    assert "HUBBARD { atomic }" in overrides.hubbard_card


# ---- write_hp_input --------------------------------------------------------


def test_write_hp_input_emits_inputhp_namelist(tmp_path: Path) -> None:
    written = write_hp_input(
        tmp_path / "hp.in",
        prefix="bulk_cu2o",
        nq=(2, 2, 2),
    )
    text = written.read_text()
    assert "&INPUTHP" in text
    assert text.rstrip().endswith("/")
    # The four parameters that matter for hp.x to find the SCF run.
    assert "prefix = 'bulk_cu2o'" in text
    assert "nq1 = 2" in text
    assert "nq2 = 2" in text
    assert "nq3 = 2" in text


def test_write_hp_input_extra_inputhp_overrides_defaults(tmp_path: Path) -> None:
    written = write_hp_input(
        tmp_path / "hp.in",
        prefix="bulk_cuo",
        extra_inputhp={"alpha_mix(1)": 0.3, "find_atpert": 2},
    )
    text = written.read_text()
    assert "alpha_mix(1) = 0.3" in text
    assert "find_atpert = 2" in text


def test_write_hp_input_creates_parent_directories(tmp_path: Path) -> None:
    written = write_hp_input(tmp_path / "deep" / "nested" / "hp.in", prefix="x")
    assert written.is_file()


# ---- fcp_overrides_for_potential (Phase 7) ---------------------------------


def test_fcp_overrides_emits_three_namelists() -> None:
    overrides = fcp_overrides_for_potential(-0.4)
    assert overrides["control"]["lfcp"] is True
    assert overrides["system"]["assume_isolated"] == "esm"
    assert overrides["system"]["esm_bc"] == "bc2"
    assert "fcp_mu" in overrides["fcp"]


def test_fcp_overrides_converts_u_she_to_fcp_mu_in_rydberg() -> None:
    """U = 0 V vs. SHE gives fcp_mu = -SHE_abs/Ry ≈ -0.326 Ry."""
    overrides = fcp_overrides_for_potential(0.0)
    expected_mu_ry = -SHE_ABSOLUTE_POTENTIAL_V / EV_PER_RYDBERG
    assert overrides["fcp"]["fcp_mu"] == pytest.approx(expected_mu_ry, rel=1e-10)


def test_fcp_overrides_higher_u_raises_fermi_level() -> None:
    """A more *positive* U vs. SHE pulls electrons OUT (lower mu_F)."""
    low = fcp_overrides_for_potential(-0.5)["fcp"]["fcp_mu"]
    high = fcp_overrides_for_potential(+0.5)["fcp"]["fcp_mu"]
    assert high < low  # more positive U → more-negative mu_F


def test_fcp_overrides_rejects_unknown_esm_bc() -> None:
    with pytest.raises(ValueError, match="Unknown esm_bc"):
        fcp_overrides_for_potential(-0.4, esm_bc="bcX")


def test_fcp_overrides_composes_with_spin_hubbard(
    tmp_path: Path, pseudo_dir: Path
) -> None:
    """Real-world combo: AFM CuO slab at applied U with DFT+U."""
    (pseudo_dir / "O.upf").write_text("")
    from copper_oxide_dft.structure_builder import build_bulk_cuo

    cuo = build_bulk_cuo()
    fcp = fcp_overrides_for_potential(-0.4)
    spin_u = spin_and_hubbard_overrides(cuo, nspin=2, hubbard_u={"Cu": 4.0})
    # FCP is a namelist-only override; the Hubbard piece is a separate card.
    merged = merge_namelist_overrides(fcp, spin_u.namelist_overrides)
    out_path = tmp_path / "cuo_fcp.in"
    write_pw_input(
        cuo,
        out_path=out_path,
        pseudopotentials={"Cu": "Cu.upf", "O": "O.upf"},
        pseudo_dir=pseudo_dir,
        extra_input_data=merged,
        additional_cards=spin_u.hubbard_card,
    )
    text = out_path.read_text()
    text_lower = text.lower()
    assert "lfcp" in text_lower
    assert "assume_isolated" in text_lower
    assert "esm" in text_lower
    assert "nspin" in text_lower
    assert "U Cu-3d 4.000000" in text
    assert "U Cu1-3d 4.000000" in text


# ---- merge_namelist_overrides + DEFAULT_PSEUDOPOTENTIALS ------------------


def test_merge_namelist_overrides_combines_dicts() -> None:
    a = {"system": {"nspin": 2}, "control": {"prefix": "x"}}
    b = {"system": {"degauss": 0.01}}
    merged = merge_namelist_overrides(a, b)
    assert merged["system"] == {"nspin": 2, "degauss": 0.01}
    assert merged["control"] == {"prefix": "x"}


def test_merge_namelist_overrides_later_overwrites_within_namelist() -> None:
    a = {"system": {"nspin": 1}}
    b = {"system": {"nspin": 2}}
    merged = merge_namelist_overrides(a, b)
    assert merged["system"]["nspin"] == 2


def test_merge_namelist_overrides_does_not_mutate_inputs() -> None:
    a = {"system": {"nspin": 2}}
    b = {"system": {"degauss": 0.01}}
    a_snap = {k: dict(v) for k, v in a.items()}
    b_snap = {k: dict(v) for k, v in b.items()}
    merge_namelist_overrides(a, b)
    assert a == a_snap
    assert b == b_snap


def test_merge_namelist_overrides_handles_none_sources() -> None:
    """None is treated as an empty mapping — lets callers splat optional dicts."""
    a = {"system": {"nspin": 2}}
    merged = merge_namelist_overrides(a, None, None)
    assert merged == {"system": {"nspin": 2}}


def test_merge_namelist_overrides_zero_sources_returns_empty() -> None:
    assert merge_namelist_overrides() == {}


def test_default_pseudopotentials_covers_project_species() -> None:
    """Cu, O, H are the three species the project actually computes with."""
    assert set(DEFAULT_PSEUDOPOTENTIALS) >= {"Cu", "O", "H"}
    for _sym, fname in DEFAULT_PSEUDOPOTENTIALS.items():
        # Filenames are PseudoDojo conventions: SymbolPart.upf at minimum.
        assert fname.endswith(".upf")
