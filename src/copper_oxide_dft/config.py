"""JSON-backed store for converged DFT parameters per system.

After a Phase 1 / Phase 2 convergence sweep lands, we lock in one set of
parameters per system (bulk Cu, Cu2O, CuO, slabs in later phases) and
re-use them across the rest of the project. Storing them in a JSON file
that lives next to the code keeps the workflow reproducible and makes
"what cutoffs did we use?" trivial to answer months later.

Schema (one entry per system)::

    {
      "schema_version": 1,
      "systems": {
        "bulk_cu":   {"ecutwfc_ry": 80.0, "kpts": [8, 8, 8],
                      "degauss_ry": 0.02, "lattice_a_ang": 3.615},
        "bulk_cu2o": {"ecutwfc_ry": 80.0, "kpts": [6, 6, 6],
                      "degauss_ry": 0.02, "hubbard_u_ev": 4.0},
        "bulk_cuo":  {"ecutwfc_ry": 80.0, "kpts": [4, 6, 4],
                      "degauss_ry": 0.02, "hubbard_u_ev": 4.0}
      }
    }

Unknown fields are preserved on round-trip so future phases can add
their own keys (e.g. ``vacuum_ang`` for slabs) without bumping the
schema version.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
"""Bump this when an incompatible change to the on-disk schema lands."""


@dataclass
class SystemConfig:
    """Converged DFT parameters for one system.

    Required fields cover what every system needs; ``extras`` carries
    anything system-specific (Hubbard U for oxides, vacuum width for
    slabs) so the dataclass doesn't have to grow indefinitely.

    Attributes:
        ecutwfc_ry: Plane-wave wavefunction cutoff (Ry).
        kpts: Monkhorst-Pack grid (kx, ky, kz).
        degauss_ry: Smearing width (Ry). Only meaningful for metals; the
            value is carried for non-metals too because changing it
            silently breaks energy comparisons across systems.
        ecutrho_ry: Charge-density cutoff. ``None`` falls back to
            ``8 * ecutwfc_ry`` at write time (PAW convention).
        extras: Free-form system-specific knobs. Keys are not validated.
    """

    ecutwfc_ry: float
    kpts: tuple[int, int, int]
    degauss_ry: float
    ecutrho_ry: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "ecutwfc_ry": self.ecutwfc_ry,
            "kpts": list(self.kpts),
            "degauss_ry": self.degauss_ry,
        }
        if self.ecutrho_ry is not None:
            out["ecutrho_ry"] = self.ecutrho_ry
        out.update(self.extras)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SystemConfig:
        known = {"ecutwfc_ry", "kpts", "degauss_ry", "ecutrho_ry"}
        kpts = data["kpts"]
        if len(kpts) != 3:
            raise ValueError(f"kpts must be 3 ints; got {kpts}")
        return cls(
            ecutwfc_ry=float(data["ecutwfc_ry"]),
            kpts=(int(kpts[0]), int(kpts[1]), int(kpts[2])),
            degauss_ry=float(data["degauss_ry"]),
            ecutrho_ry=(
                None if data.get("ecutrho_ry") is None else float(data["ecutrho_ry"])
            ),
            extras={k: v for k, v in data.items() if k not in known},
        )


@dataclass
class ProjectConfig:
    """Top-level container: one :class:`SystemConfig` per system name."""

    systems: dict[str, SystemConfig] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "systems": {name: sc.to_dict() for name, sc in self.systems.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectConfig:
        version = int(data.get("schema_version", SCHEMA_VERSION))
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported config schema_version={version}; "
                f"this code understands {SCHEMA_VERSION}."
            )
        systems = {
            name: SystemConfig.from_dict(sys_data)
            for name, sys_data in data.get("systems", {}).items()
        }
        return cls(systems=systems)


def load_config(path: str | Path) -> ProjectConfig:
    """Load a :class:`ProjectConfig` from a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed config object.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the schema version is unsupported.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file does not exist: {p}")
    return ProjectConfig.from_dict(json.loads(p.read_text()))


def save_config(config: ProjectConfig, path: str | Path) -> Path:
    """Serialize a :class:`ProjectConfig` to a JSON file.

    Args:
        config: Config to write.
        path: Destination path. Parent directories are created.

    Returns:
        Absolute path to the written file.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(config.to_dict(), indent=2))
    return p.resolve()


__all__ = (
    "SCHEMA_VERSION",
    "ProjectConfig",
    "SystemConfig",
    "load_config",
    "save_config",
)
