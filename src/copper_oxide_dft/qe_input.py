"""Quantum ESPRESSO ``pw.x`` input file generation with project defaults.

Wraps :func:`ase.io.espresso.write_espresso_in` with the project's standard
choices (Marzari-Vanderbilt smearing, PBE+U-ready namelist structure,
sensible SCF convergence thresholds).

Pseudopotentials are looked up relative to ``pseudo_dir``. If not passed
explicitly, the environment variable ``CUOXDFT_PSEUDO_DIR`` is used; the
function raises :class:`FileNotFoundError` if neither resolves to an
existing directory.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.io.espresso import write_espresso_in

PSEUDO_DIR_ENV_VAR = "CUOXDFT_PSEUDO_DIR"

DEFAULT_ECUTWFC_RY = 80.0
"""Wavefunction plane-wave cutoff. Conservative starting point; refined in
the Phase 1 convergence sweep."""

DEFAULT_DEGAUSS_RY = 0.02
"""Smearing width for Marzari-Vanderbilt cold smearing. Required for metallic Cu;
see docs/ground_truths.md (Cu oxide DFT gotchas)."""

DEFAULT_HUBBARD_U_CU_3D_EV = 4.0
"""Default Hubbard U on Cu 3d (eV). Literature pick (Mosey/Carter ~4 eV) for the
Phase 4 Pourbaix work; calibrate via hp.x linear response in Phase 2."""

SUPPORTED_CALCULATIONS = frozenset({"scf", "nscf", "relax", "vc-relax", "md"})


def _resolve_pseudo_dir(pseudo_dir: str | os.PathLike[str] | None) -> Path:
    if pseudo_dir is None:
        env = os.environ.get(PSEUDO_DIR_ENV_VAR)
        if not env:
            raise FileNotFoundError(
                f"Pseudopotential directory not provided and ${PSEUDO_DIR_ENV_VAR}"
                " is unset. Pass `pseudo_dir=` or set the environment variable."
            )
        pseudo_dir = env
    resolved = Path(pseudo_dir).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Pseudopotential directory does not exist: {resolved}")
    return resolved


def write_pw_input(
    atoms: Atoms,
    out_path: str | os.PathLike[str],
    pseudopotentials: Mapping[str, str],
    *,
    calculation: str = "scf",
    prefix: str = "calc",
    ecutwfc: float = DEFAULT_ECUTWFC_RY,
    ecutrho: float | None = None,
    kpts: tuple[int, int, int] = (8, 8, 8),
    koffset: tuple[int, int, int] = (0, 0, 0),
    degauss: float = DEFAULT_DEGAUSS_RY,
    pseudo_dir: str | os.PathLike[str] | None = None,
    extra_input_data: Mapping[str, Mapping[str, Any]] | None = None,
) -> Path:
    """Write a ``pw.x`` input file with project-standard defaults.

    Args:
        atoms: Structure to compute. Cell and positions are written directly;
            ``ibrav=0`` is used so QE consumes the supplied vectors.
        out_path: File path to write the input to. Parent directories are
            created if needed.
        pseudopotentials: Mapping from chemical symbol to UPF filename
            (e.g. ``{"Cu": "Cu.upf"}``). Files must live in ``pseudo_dir``.
        calculation: pw.x calculation type. Must be one of
            :data:`SUPPORTED_CALCULATIONS`. ``"vc-relax"`` and ``"relax"``
            automatically populate the IONS / CELL namelists with BFGS defaults.
        prefix: ``CONTROL.prefix`` value (used by QE for output filenames).
        ecutwfc: Plane-wave wavefunction cutoff (Ry).
        ecutrho: Charge-density cutoff (Ry). Defaults to ``8 * ecutwfc``,
            which is appropriate for PAW pseudopotentials.
        kpts: Monkhorst-Pack k-point grid.
        koffset: Grid offset (use ``(1, 1, 1)`` for shifted, gamma-excluded).
        degauss: Smearing width (Ry).
        pseudo_dir: Directory containing UPF files. Falls back to
            ``$CUOXDFT_PSEUDO_DIR`` if ``None``.
        extra_input_data: Optional namelist overrides merged on top of
            defaults (e.g. ``{"system": {"nspin": 2}}``).

    Returns:
        Path to the written input file.

    Raises:
        ValueError: If ``calculation`` is not in :data:`SUPPORTED_CALCULATIONS`.
        FileNotFoundError: If ``pseudo_dir`` and ``$CUOXDFT_PSEUDO_DIR`` are
            both missing or do not point to an existing directory.
    """
    if calculation not in SUPPORTED_CALCULATIONS:
        raise ValueError(
            f"Unsupported calculation={calculation!r}; "
            f"expected one of {sorted(SUPPORTED_CALCULATIONS)}"
        )
    pseudo_dir_resolved = _resolve_pseudo_dir(pseudo_dir)

    if ecutrho is None:
        ecutrho = 8.0 * ecutwfc

    input_data: dict[str, dict[str, Any]] = {
        "control": {
            "calculation": calculation,
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
    if calculation in ("relax", "vc-relax", "md"):
        input_data["ions"] = {"ion_dynamics": "bfgs"}
    if calculation == "vc-relax":
        input_data["cell"] = {"cell_dynamics": "bfgs", "press": 0.0}

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


def spin_and_hubbard_overrides(
    atoms: Atoms,
    *,
    nspin: int = 2,
    hubbard_u: Mapping[str, float] | None = None,
    starting_magnetization: Mapping[str, float] | Sequence[float] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build the ``extra_input_data`` override for spin-polarized / DFT+U runs.

    Quantum ESPRESSO needs three things to do a magnetic DFT+U calculation
    correctly, and they go into three different namelists. This helper
    assembles them so callers don't have to remember the syntax:

    * ``SYSTEM.nspin = 2`` — turn on spin polarization (mandatory for CuO,
      O2, and any O-containing slab).
    * ``SYSTEM.starting_magnetization(i)`` — initial moment per species
      index. The right starting guess matters: zero moments often produce
      a non-magnetic (wrong) ground state for CuO.
    * ``SYSTEM.Hubbard_U(i)`` — the U value per species. PBE alone gives a
      qualitatively wrong band gap and lattice for Cu oxides; DFT+U fixes
      both. The project default is ~4 eV on Cu 3d (see ground_truths.md).

    Species index assignment mirrors ASE's QE writer. With nspin=2 (or any
    nonzero initial magmom on ``atoms``), ASE splits atoms of the same
    chemical symbol into separate species when their initial magnetic
    moments differ — e.g. CuO's AFM ordering produces species "Cu" (mag
    +1) and "Cu1" (mag -1), both still Cu chemically, both needing the
    same Hubbard U. This helper emits ``Hubbard_U(i)`` for every species
    whose chemical symbol appears in ``hubbard_u``, including the
    AFM-split duplicates.

    For per-atom AFM starting moments (alternating +/- on Cu in CuO), set
    ``atoms.set_initial_magnetic_moments(...)`` BEFORE calling
    :func:`write_pw_input`; ASE will write per-species
    ``starting_magnetization(i)`` cards derived from those values and
    will overwrite any ``starting_magnetization`` you pass through this
    helper. The :func:`build_bulk_cuo` builder sets the AFM moments
    automatically.

    Args:
        atoms: Structure being computed. Used to determine the species
            list (chemical symbols + per-atom magnetic moments).
        nspin: 1 (non-magnetic) or 2 (collinear spin-polarized).
        hubbard_u: Mapping from chemical symbol to U value in eV. Species
            not present in the structure are silently ignored.
        starting_magnetization: Mapping from symbol to initial moment, OR
            a sequence indexed by species order. Use this only when
            ``atoms`` has no per-atom magmoms set (otherwise ASE
            overrides). For AFM, set per-atom magmoms on ``atoms``.

    Returns:
        A namelist-override dict suitable for the ``extra_input_data``
        argument of :func:`write_pw_input`.

    Example:
        >>> from copper_oxide_dft.structure_builder import (
        ...     build_bulk_cu2o, build_bulk_cuo
        ... )
        >>> # Non-magnetic Cu2O: one Cu species, U on Cu.
        >>> overrides = spin_and_hubbard_overrides(
        ...     build_bulk_cu2o(), nspin=1, hubbard_u={"Cu": 4.0}
        ... )
        >>> overrides["system"]["nspin"]
        1
        >>> overrides["system"]["Hubbard_U(1)"]
        4.0
        >>> # AFM CuO: two Cu species (the +/- sublattices), U on BOTH.
        >>> overrides = spin_and_hubbard_overrides(
        ...     build_bulk_cuo(), nspin=2, hubbard_u={"Cu": 4.0}
        ... )
        >>> overrides["system"]["Hubbard_U(1)"]
        4.0
        >>> overrides["system"]["Hubbard_U(2)"]
        4.0
    """
    if nspin not in (1, 2):
        raise ValueError(f"nspin must be 1 or 2, got {nspin}")

    species_pairs = _ase_species_breakdown(atoms, nspin=nspin)

    system: dict[str, Any] = {"nspin": nspin}

    if hubbard_u:
        for i, (_, symbol) in enumerate(species_pairs, start=1):
            if symbol in hubbard_u:
                system[f"Hubbard_U({i})"] = float(hubbard_u[symbol])

    if starting_magnetization is not None:
        if isinstance(starting_magnetization, Mapping):
            for i, (_, symbol) in enumerate(species_pairs, start=1):
                if symbol in starting_magnetization:
                    system[f"starting_magnetization({i})"] = float(
                        starting_magnetization[symbol]
                    )
        else:
            for i, mag in enumerate(starting_magnetization, start=1):
                system[f"starting_magnetization({i})"] = float(mag)

    return {"system": system}


SHE_ABSOLUTE_POTENTIAL_V = 4.44
"""Absolute potential of the Standard Hydrogen Electrode vs. vacuum (V).

Used to convert a target U vs. SHE into a Fermi-level chemical potential
suitable for FCP. The Trasatti value (W. Schmickler & E. Santos,
*Interfacial Electrochemistry*, 2010) is 4.44 V; some references use
4.28–4.60 V. Override per-project if a different convention is needed.
"""

EV_PER_RYDBERG = 13.605693122994
"""Rydberg-to-eV conversion factor (CODATA 2018). Used by the FCP helper
because QE's ``fcp_mu`` is expressed in Rydberg internally."""


def fcp_overrides_for_potential(
    u_she_v: float,
    *,
    esm_bc: str = "bc2",
    esm_w_ang: float = 0.0,
    fcp_dynamics: str = "lm",
    fcp_thr_ev: float = 1.0e-2,
    she_absolute_v: float = SHE_ABSOLUTE_POTENTIAL_V,
) -> dict[str, dict[str, Any]]:
    """Build the namelist override that sets up ESM + FCP at a target U.

    Wires three things together for constant-potential DFT with QE's
    Effective Screening Medium (ESM) and Fictitious Charge Potentiostat
    (FCP):

    1. ``&CONTROL``: ``calculation = 'scf'`` is unchanged; we add
       ``lfcp = .true.`` so the FCP loop runs after each SCF.
    2. ``&SYSTEM``: ``assume_isolated = 'esm'`` activates the ESM
       boundary conditions, ``esm_bc`` chooses symmetric vs. asymmetric.
    3. ``&FCP``: target chemical potential, dynamics, and convergence
       threshold.

    The target U vs. SHE is converted to a Fermi-level chemical
    potential via::

        mu_F (eV vs. vacuum) = -(SHE_absolute + U_SHE)
        fcp_mu (Ry)         = mu_F / 13.605693

    so applying U = -0.4 V vs. SHE gives mu_F = -(4.44 + (-0.4)) = -4.04
    eV, and fcp_mu ≈ -0.297 Ry. This is the QE convention for FCP
    (chemical potential of electrons, vacuum-referenced, in Ry).

    Args:
        u_she_v: Target electrode potential (V vs. SHE).
        esm_bc: ESM boundary condition. ``"bc1"`` = vacuum on both sides
            (symmetric slab); ``"bc2"`` = vacuum on one side, metal
            counter-electrode on the other (the standard for
            electrochemistry).
        esm_w_ang: Width of additional vacuum region inserted by ESM
            inside the cell (Å). Usually 0 unless the cell is too small.
        fcp_dynamics: FCP step algorithm. ``"lm"`` = line minimization
            (robust default); ``"newton"`` is faster when close to
            convergence.
        fcp_thr_ev: FCP convergence threshold for chemical-potential
            mismatch (eV).
        she_absolute_v: Absolute potential of SHE vs. vacuum (V). The
            Trasatti value 4.44 is the default; override if your group
            uses a different convention.

    Returns:
        Namelist-override dict suitable for the ``extra_input_data``
        argument of :func:`write_pw_input`.

    Example:
        >>> # Set up a Cu(111) slab at U = -0.4 V vs. SHE.
        >>> overrides = fcp_overrides_for_potential(-0.4)
        >>> overrides["control"]["lfcp"]
        True
        >>> overrides["system"]["assume_isolated"]
        'esm'
        >>> overrides["system"]["esm_bc"]
        'bc2'
    """
    if esm_bc not in {"bc1", "bc2", "bc3", "bc4"}:
        raise ValueError(
            f"Unknown esm_bc {esm_bc!r}; expected one of bc1, bc2, bc3, bc4."
        )

    fermi_level_ev_vs_vacuum = -(she_absolute_v + u_she_v)
    fcp_mu_ry = fermi_level_ev_vs_vacuum / EV_PER_RYDBERG

    return {
        "control": {
            "lfcp": True,
        },
        "system": {
            "assume_isolated": "esm",
            "esm_bc": esm_bc,
            "esm_w": esm_w_ang,
        },
        "fcp": {
            "fcp_mu": fcp_mu_ry,
            "fcp_dynamics": fcp_dynamics,
            "fcp_thr": fcp_thr_ev,
        },
    }


def write_hp_input(
    out_path: str | os.PathLike[str],
    *,
    prefix: str = "calc",
    nq: tuple[int, int, int] = (2, 2, 2),
    iverbosity: int = 1,
    conv_thr_chpsi: float = 1.0e-12,
    extra_inputhp: Mapping[str, Any] | None = None,
) -> Path:
    """Write a ``hp.x`` input file for self-consistent Hubbard U linear response.

    ``hp.x`` is QE's density-functional perturbation tool for Hubbard
    parameters: it computes the response of the Hubbard occupations to
    a small perturbation and back-solves the self-consistent U value
    that reproduces that response. The result replaces the literature
    U we currently use as a default.

    The standard recipe (per the QE / hp.x manual): run an SCF first
    with some starting U (the ``prefix`` here must match the SCF run's
    ``CONTROL.prefix`` so hp.x finds the saved wavefunctions). Then
    point hp.x at that prefix with a Monkhorst-Pack q-grid; a 2x2x2
    grid is usually enough for small bulk cells.

    Args:
        out_path: File path to write the input to. Convention: place
            ``hp.in`` in the same directory as the parent SCF's ``pw.in``.
        prefix: Must match the SCF ``prefix``. hp.x reads the saved
            wavefunctions/Hubbard occupations from there.
        nq: Monkhorst-Pack q-grid for the perturbation. Increase for
            larger cells where the response is longer-ranged.
        iverbosity: hp.x verbosity. ``1`` is the standard "tell me what's
            going on without flooding the log" level.
        conv_thr_chpsi: SCF threshold for the linear-response charge
            perturbation (Ry).
        extra_inputhp: Optional overrides merged into the ``INPUTHP``
            namelist (e.g. ``{"alpha_mix(1)": 0.3}``).

    Returns:
        Path to the written input file.
    """
    inputhp: dict[str, Any] = {
        "prefix": prefix,
        "outdir": "./tmp",
        "nq1": nq[0],
        "nq2": nq[1],
        "nq3": nq[2],
        "iverbosity": iverbosity,
        "conv_thr_chpsi": conv_thr_chpsi,
    }
    if extra_inputhp:
        inputhp.update(extra_inputhp)

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    lines = ["&INPUTHP"]
    for key, value in inputhp.items():
        lines.append(f"  {key} = {_format_namelist_value(value)}")
    lines.append("/")
    p.write_text("\n".join(lines) + "\n")
    return p


def _format_namelist_value(value: Any) -> str:
    """Format a Python value as a Fortran namelist literal."""
    if isinstance(value, bool):
        return ".true." if value else ".false."
    if isinstance(value, str):
        return f"'{value}'"
    return str(value)


def _ase_species_breakdown(
    atoms: Atoms, *, nspin: int
) -> list[tuple[float, str]]:
    """Mirror ASE's QE-writer species-splitting algorithm.

    Returns a list of ``(magmom, symbol)`` tuples in the order ASE will
    assign species indices. With ``nspin=2`` (or any nonzero initial
    magmom) ASE creates a new species for each unique ``(symbol, magmom)``
    pair; otherwise it keys only on chemical symbol.

    See ase.io.espresso.write_espresso_in for the reference algorithm.
    """
    magmoms = atoms.get_initial_magnetic_moments()
    magnetic = nspin == 2 or any(m != 0.0 for m in magmoms)
    seen: dict[tuple[str, float], None] = {}
    ordered: list[tuple[float, str]] = []
    for symbol, magmom in zip(atoms.get_chemical_symbols(), magmoms, strict=True):
        key: tuple[str, float] = (
            (symbol, float(magmom)) if magnetic else (symbol, 0.0)
        )
        if key not in seen:
            seen[key] = None
            ordered.append((float(magmom), symbol))
    return ordered
