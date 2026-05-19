"""ESM-FCP rerank of the top-K GCGA candidates at constant U.

After :mod:`copper_oxide_dft.ml.ensemble` produces the per-x_O minimum-Ω
phases (Block E of :doc:`/docs/ml-gcgo-pivot.md`), the top-K of those
are re-relaxed at a fixed electrochemical potential to answer the
project's actual question: "what is the predicted surface at U = −0.8 V
vs Ag/AgCl?"

The ESM-FCP relaxation is run on Frontier (the GPU-accelerated AMD
build of QE). This module:

1. Writes one ``candidate_NN/pw.in`` per top-K candidate, configured
   for ESM-FCP at the target U via the existing
   :func:`copper_oxide_dft.qe_input.fcp_overrides_for_potential`.
2. Wraps each with a Frontier SLURM script via the existing
   :func:`copper_oxide_dft.submit.write_slurm_scripts_for_tree`.
3. After the user submits + reaps the runs, parses the FCP-converged
   ``tot_charge`` from each ``pw.out``, computes the grand-canonical
   ``Ω(U)``, and ranks the candidates.

The constant-U grand potential is

    Ω(U) = E_DFT − μ_e · N_e_excess,    μ_e = −(V_abs + U)

where ``N_e_excess`` is the electron count relative to the neutral cell.
QE reports the FCP-converged ``tot_charge`` (positive = electrons
removed), so ``N_e_excess = −tot_charge``, and

    Ω(U) = E_DFT + μ_e · tot_charge.

This is the relation the manuscript's Phase-7 helper buried in the
startup doc; we make it a first-class function here so the answer is
reproducible from any (pw.in, pw.out) pair.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ase import Atoms

from copper_oxide_dft.config import SystemConfig
from copper_oxide_dft.ml.ensemble import Phase
from copper_oxide_dft.parse import parse_pw_output
from copper_oxide_dft.qe_input import (
    DEFAULT_HUBBARD_U_CU_3D_EV,
    DEFAULT_PSEUDOPOTENTIALS,
    fcp_overrides_for_potential,
    merge_namelist_overrides,
    spin_and_hubbard_overrides,
    write_pw_input,
)
from copper_oxide_dft.submit import SlurmConfig, write_slurm_scripts_for_tree

DEFAULT_AG_AGCL_ABSOLUTE_POTENTIAL_V = 4.64
"""Absolute potential of the Ag/AgCl reference electrode in vacuum-referenced
DFT (= 4.44 V SHE + 0.197 V Ag/AgCl sat-KCl). See
:doc:`/docs/ground_truths.md` 2026-05-18 (Real experimental system)."""

DEFAULT_TARGET_POTENTIAL_V = -0.8
"""Target electrochemical potential for the rerank: U = −0.8 V vs Ag/AgCl."""


_TOT_CHARGE_RE = re.compile(
    r"(?<![A-Za-z_])tot_charge\s*=\s*([-+0-9.eEdD]+)", re.IGNORECASE
)
"""Last standalone ``tot_charge = X`` occurrence in pw.out is the
FCP-converged charge. The negative-look-behind prevents matching
prefixed variants (e.g. ``new_tot_charge``). The "last occurrence"
heuristic relies on QE printing the converged value after the input
namelist echo — verify against the first real Frontier pw.out before
trusting Ω(U) for production rankings."""


def prepare_fcp_inputs(
    candidates: Sequence[Phase],
    out_root: str | os.PathLike[str],
    *,
    system_config: SystemConfig,
    u_target_v: float = DEFAULT_TARGET_POTENTIAL_V,
    reference_absolute_v: float = DEFAULT_AG_AGCL_ABSOLUTE_POTENTIAL_V,
    hubbard_u_ev: float = DEFAULT_HUBBARD_U_CU_3D_EV,
    pseudopotentials: dict[str, str] | None = None,
    pseudo_dir: str | os.PathLike[str] | None = None,
    kpts: tuple[int, int, int] = (1, 1, 1),
    calculation: str = "relax",
) -> list[Path]:
    """Write per-candidate ``pw.in`` files for ESM-FCP at constant U.

    Args:
        candidates: Top-K phases from the GCGA ensemble.
        out_root: Directory to populate. Each candidate gets its own
            subdirectory ``candidate_NN`` containing one ``pw.in``.
        system_config: Phase 1 :class:`SystemConfig` (``bulk_cu``).
            Provides ``ecutwfc_ry``, ``degauss_ry``.
        u_target_v: Target electrochemical potential (V vs the
            reference electrode in ``reference_absolute_v``).
        reference_absolute_v: Absolute potential of the reference
            electrode in vacuum (V). Default = Ag/AgCl (4.64 V).
        hubbard_u_ev: Hubbard U on Cu 3d.
        pseudopotentials: Optional override of the species → UPF mapping.
            Defaults to ``{"Cu": "Cu.upf", "O": "O.upf"}``.
        pseudo_dir: Directory containing the UPF files. Falls back to
            ``$CUOXDFT_PSEUDO_DIR``.
        kpts: Monkhorst-Pack grid. Γ-only is appropriate for the
            ~200-atom GCGA candidates; bump to e.g. (2, 2, 1) for
            smaller slabs.
        calculation: ``"relax"`` (positions only) or ``"scf"`` (single
            point). vc-relax is *not* recommended with ESM-FCP — the
            screening-medium boundary condition assumes a fixed cell.

    Returns:
        Paths to the written ``pw.in`` files, one per candidate.

    Raises:
        ValueError: If ``candidates`` is empty.
    """
    if not candidates:
        raise ValueError("candidates is empty — nothing to rerank.")
    if pseudopotentials is None:
        pseudopotentials = DEFAULT_PSEUDOPOTENTIALS

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    fcp = fcp_overrides_for_potential(
        u_target_v, she_absolute_v=reference_absolute_v
    )

    written: list[Path] = []
    for idx, candidate in enumerate(candidates):
        atoms = candidate.atoms
        spin = spin_and_hubbard_overrides(
            atoms, nspin=2, hubbard_u={"Cu": hubbard_u_ev}
        )
        merged = merge_namelist_overrides(fcp, spin)

        sample_dir = out_root / f"candidate_{idx:02d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        path = write_pw_input(
            atoms,
            out_path=sample_dir / "pw.in",
            pseudopotentials=pseudopotentials,
            calculation=calculation,
            prefix=f"candidate_{idx:02d}",
            ecutwfc=system_config.ecutwfc_ry,
            kpts=kpts,
            degauss=system_config.degauss_ry,
            pseudo_dir=pseudo_dir,
            extra_input_data=merged,
        )
        written.append(path)
    return written


def write_frontier_submit_scripts(
    out_root: str | os.PathLike[str],
    account: str,
    *,
    walltime: str = "4:00:00",
    qe_module: str = "quantum-espresso/7.3-gpu",
) -> list[Path]:
    """Wrap every ``pw.in`` under ``out_root`` with a Frontier SLURM script.

    Thin shim around :func:`copper_oxide_dft.submit.write_slurm_scripts_for_tree`
    + :meth:`SlurmConfig.for_frontier`, kept here so a caller can do the
    whole rerank in two function calls.

    Args:
        out_root: Directory populated by :func:`prepare_fcp_inputs`.
        account: SLURM allocation (``-A`` argument).
        walltime: ``-t`` argument. ESM-FCP on 200-atom cells is slower
            than the dataset relaxations; 4 hours is the safe default.
        qe_module: Module name on Frontier. Verify against
            ``module avail quantum-espresso``.

    Returns:
        Paths to the written ``submit.sh`` files.
    """
    cfg = SlurmConfig.for_frontier(account=account, walltime=walltime, qe_module=qe_module)
    return write_slurm_scripts_for_tree(Path(out_root), cfg)


def parse_fcp_tot_charge(pw_out_path: str | os.PathLike[str]) -> float | None:
    """Return the last ``tot_charge`` value in a pw.out, or None if absent.

    QE's FCP loop prints the converged ``tot_charge`` near the end of
    the output. Positive = electrons removed (oxidised); negative =
    electrons added (reduced).

    Args:
        pw_out_path: Path to a captured pw.x stdout.

    Returns:
        Converged tot_charge in electrons (float), or None if no match.
    """
    text = Path(pw_out_path).read_text()
    matches = _TOT_CHARGE_RE.findall(text)
    if not matches:
        return None
    return float(matches[-1].replace("D", "e").replace("d", "e"))


def grand_potential_at_u(
    energy_ev: float,
    tot_charge: float,
    *,
    u_target_v: float = DEFAULT_TARGET_POTENTIAL_V,
    reference_absolute_v: float = DEFAULT_AG_AGCL_ABSOLUTE_POTENTIAL_V,
) -> float:
    """Constant-U grand potential.

    ``Ω(U) = E_DFT − μ_e · N_e_excess`` with ``μ_e = −(V_abs + U)`` and
    ``N_e_excess = −tot_charge``. So:

        Ω(U) = E_DFT + μ_e · tot_charge.

    Args:
        energy_ev: Total DFT energy of the FCP-converged structure (eV).
        tot_charge: Converged ``tot_charge`` from the FCP loop.
        u_target_v: Electrochemical potential the run was held at.
        reference_absolute_v: Absolute potential of the reference
            electrode in vacuum.

    Returns:
        Grand potential at U (eV).
    """
    mu_e_ev_vs_vacuum = -(reference_absolute_v + u_target_v)
    return float(energy_ev) + mu_e_ev_vs_vacuum * float(tot_charge)


@dataclass(frozen=True)
class FcpRerankResult:
    """One row of the constant-U rerank table.

    Attributes:
        candidate_id: ``candidate_NN`` directory name.
        atoms: FCP-converged structure (from pw.out, if available).
        energy_ev: Converged DFT total energy.
        tot_charge: Converged FCP charge (electrons; ``None`` if absent).
        omega_u_ev: Constant-U grand potential, or ``None`` if
            ``tot_charge`` was missing.
        u_target_v: Target potential the run was held at.
    """

    candidate_id: str
    atoms: Atoms | None
    energy_ev: float
    tot_charge: float | None
    omega_u_ev: float | None
    u_target_v: float


def rank_fcp_results(
    out_root: str | os.PathLike[str],
    *,
    u_target_v: float = DEFAULT_TARGET_POTENTIAL_V,
    reference_absolute_v: float = DEFAULT_AG_AGCL_ABSOLUTE_POTENTIAL_V,
) -> list[FcpRerankResult]:
    """Walk ``out_root/candidate_NN/pw.out`` and rank by Ω(U).

    Candidates whose FCP loop did not converge (``tot_charge`` absent
    or the run errored out) appear at the *end* of the returned list
    with ``omega_u_ev = None`` so they're easy to triage.

    Args:
        out_root: Directory populated by :func:`prepare_fcp_inputs` and
            then run on Frontier.
        u_target_v: U the runs were held at (must match
            :func:`prepare_fcp_inputs`).
        reference_absolute_v: Absolute potential of the reference
            electrode.

    Returns:
        List of :class:`FcpRerankResult`, sorted by Ω(U) ascending,
        with non-converged runs trailing.
    """
    out_root = Path(out_root)
    results: list[FcpRerankResult] = []
    for candidate_dir in sorted(out_root.glob("candidate_*")):
        pw_out = candidate_dir / "pw.out"
        if not pw_out.is_file():
            continue

        try:
            scalars = parse_pw_output(pw_out)
        except ValueError:
            results.append(
                FcpRerankResult(
                    candidate_id=candidate_dir.name,
                    atoms=None,
                    energy_ev=float("nan"),
                    tot_charge=None,
                    omega_u_ev=None,
                    u_target_v=u_target_v,
                )
            )
            continue

        tot_charge = parse_fcp_tot_charge(pw_out)
        omega = (
            grand_potential_at_u(
                scalars.total_energy_ev,
                tot_charge,
                u_target_v=u_target_v,
                reference_absolute_v=reference_absolute_v,
            )
            if tot_charge is not None
            else None
        )

        atoms = _read_final_geometry(pw_out)

        results.append(
            FcpRerankResult(
                candidate_id=candidate_dir.name,
                atoms=atoms,
                energy_ev=scalars.total_energy_ev,
                tot_charge=tot_charge,
                omega_u_ev=omega,
                u_target_v=u_target_v,
            )
        )

    converged = [r for r in results if r.omega_u_ev is not None]
    unconverged = [r for r in results if r.omega_u_ev is None]
    converged.sort(key=lambda r: r.omega_u_ev)
    return converged + unconverged


def _read_final_geometry(pw_out: Path) -> Atoms | None:
    """Best-effort: ASE QE reader. Returns ``None`` on any failure."""
    from ase.io import read as ase_read

    try:
        return ase_read(str(pw_out), format="espresso-out")
    except Exception:  # noqa: BLE001
        return None
