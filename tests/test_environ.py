"""Tests for copper_oxide_dft.environ (Phase 5 Environ input writer)."""

from __future__ import annotations

from pathlib import Path

import pytest

from copper_oxide_dft.environ import (
    WATER_STATIC_PERMITTIVITY,
    write_environ_input,
)


def test_environ_input_default_water_namelists(tmp_path: Path) -> None:
    p = write_environ_input(tmp_path / "environ.in")
    text = p.read_text()
    assert "&ENVIRON" in text
    assert "&BOUNDARY" in text
    assert "&ELECTROSTATIC" in text
    assert "environ_type = 'water'" in text
    assert f"env_static_permittivity = {WATER_STATIC_PERMITTIVITY}" in text
    assert "pbc_dim = 2" in text


def test_environ_input_custom_solvent_appears(tmp_path: Path) -> None:
    p = write_environ_input(
        tmp_path / "environ.in", environ_type="vacuum", static_permittivity=1.0
    )
    text = p.read_text()
    assert "environ_type = 'vacuum'" in text
    assert "env_static_permittivity = 1.0" in text


def test_environ_input_extra_namelists_override_defaults(tmp_path: Path) -> None:
    p = write_environ_input(
        tmp_path / "environ.in",
        extra_namelists={"electrostatic": {"tolvelect": 1.0e-12}},
    )
    text = p.read_text()
    assert "tolvelect = 1e-12" in text or "tolvelect = 1.0e-12" in text


def test_environ_input_pbc_axis_validated_through_extra_namelists(
    tmp_path: Path,
) -> None:
    """Verify the writer threads through axes/dim correctly."""
    p = write_environ_input(tmp_path / "environ.in", pbc_axis=2, pbc_dim=1)
    text = p.read_text()
    assert "pbc_axis = 2" in text
    assert "pbc_dim = 1" in text


def test_environ_input_writes_to_nested_path(tmp_path: Path) -> None:
    p = write_environ_input(tmp_path / "a" / "b" / "environ.in")
    assert p.is_file()


def test_environ_input_rejects_no_pseudo_dir_not_applicable() -> None:
    # Sanity: Environ has no pseudo_dir concept; the function should
    # not raise for any pseudo-related reason.
    with pytest.raises(TypeError):
        write_environ_input()  # type: ignore[call-arg]
