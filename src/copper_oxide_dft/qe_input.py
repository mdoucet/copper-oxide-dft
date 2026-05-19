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
from dataclasses import dataclass, field
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

DEFAULT_HUBBARD_PROJECTOR_TYPE = "atomic"
"""QE 7.1+ HUBBARD-card projector type. ``atomic`` reproduces the
``lda_plus_u_kind=0`` default that older literature U values
(Mosey/Carter 4 eV on Cu 3d) were derived against. Switching to
``ortho-atomic`` (QE's current recommendation for new calculations)
shifts the effective U by typically 0.5–1.0 eV — do not flip the
default without re-doing the hp.x calibration."""

DEFAULT_HUBBARD_MANIFOLDS: Mapping[str, str] = {
    "Cu": "3d",
    "Fe": "3d",
    "Co": "3d",
    "Ni": "3d",
    "Mn": "3d",
    "O": "2p",
}
"""Default ``{symbol: manifold}`` for the QE 7.1+ HUBBARD card. The
``HUBBARD`` card requires an explicit manifold (e.g. ``Cu-3d``) per
species; this map keeps callers from having to spell it out for the
common transition metals and O. Add new species here when needed."""

SUPPORTED_CALCULATIONS = frozenset({"scf", "nscf", "relax", "vc-relax", "md"})

DEFAULT_PSEUDOPOTENTIALS: Mapping[str, str] = {
    "Cu": "Cu.upf",
    "O": "O.upf",
    "H": "H.upf",
}
"""Project-standard PseudoDojo PBE PAW filenames. The ESM-FCP and MLIP-GCGO
pipelines both consume this; promote any additional species here rather than
duplicating the dict at call sites."""


def merge_namelist_overrides(
    *sources: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """Merge multiple ``extra_input_data`` dicts, last-wins per key.

    The :func:`write_pw_input` ``extra_input_data`` parameter takes a
    ``{namelist: {key: value}}`` mapping. Two helpers in this module
    (:func:`fcp_overrides_for_potential`, :func:`spin_and_hubbard_overrides`)
    each return such a mapping (the latter via the
    ``.namelist_overrides`` attribute of :class:`SpinHubbardOverrides`),
    and downstream callers (ML dataset generation, ESM-FCP rerank)
    routinely need to combine them. This helper does the per-namelist
    ``update(...)`` so callers don't re-implement it.

    Args:
        *sources: Zero or more ``{namelist: {key: value}}`` mappings.
            ``None`` is silently treated as the empty mapping so
            callers can splat in optional dicts. Note that the Hubbard
            U card is **not** in any namelist (QE 7.1+ moved it out);
            pass :attr:`SpinHubbardOverrides.hubbard_card` to
            :func:`write_pw_input` via ``additional_cards=`` instead.

    Returns:
        A new ``dict[str, dict[str, Any]]`` containing the merged
        overrides. Inputs are not mutated.

    Example:
        >>> fcp = fcp_overrides_for_potential(-0.8, she_absolute_v=4.64)
        >>> spin = spin_and_hubbard_overrides(atoms, nspin=2, hubbard_u={"Cu": 4.0})
        >>> merged = merge_namelist_overrides(fcp, spin.namelist_overrides)
        >>> write_pw_input(
        ...     atoms, ...,
        ...     extra_input_data=merged,
        ...     additional_cards=spin.hubbard_card,
        ... )
    """
    merged: dict[str, dict[str, Any]] = {}
    for source in sources:
        if source is None:
            continue
        for namelist, entries in source.items():
            merged.setdefault(namelist, {}).update(entries)
    return merged


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
    additional_cards: str | None = None,
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
        additional_cards: Free-form text appended after QE's namelist /
            card sections. Use for the QE 7.1+ ``HUBBARD`` card —
            :func:`spin_and_hubbard_overrides` returns one in its
            ``hubbard_card`` attribute. Pass ``None`` (default) for a
            run without Hubbard U.

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
    # Empty strings would be appended verbatim by ASE; normalise to None.
    cards = additional_cards if additional_cards else None
    with out_path.open("w") as fh:
        write_espresso_in(
            fh,
            atoms,
            input_data=input_data,
            pseudopotentials=dict(pseudopotentials),
            kpts=kpts,
            koffset=koffset,
            additional_cards=cards,
        )
    return out_path


@dataclass(frozen=True)
class SpinHubbardOverrides:
    """Spin polarization + Hubbard U settings, ready for :func:`write_pw_input`.

    Two-piece return because the QE 7.1+ HUBBARD card is no longer a
    namelist key — it's a separate post-namelist card. Pass
    ``namelist_overrides`` to ``write_pw_input(..., extra_input_data=...)``
    and ``hubbard_card`` to ``write_pw_input(..., additional_cards=...)``.

    Attributes:
        namelist_overrides: ``{"system": {"nspin": ..., "starting_magnetization(i)": ...}}``.
            Does NOT contain ``Hubbard_U(i)`` keys (QE 7.1 removed those —
            the writer would raise ``DFT+Hubbard input syntax has changed``).
        hubbard_card: New-syntax HUBBARD card text, ready to drop in
            as-is. Empty string when no Hubbard U was requested.
    """

    namelist_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    hubbard_card: str = ""


def _ase_species_labels(species_pairs: Sequence[tuple[float, str]]) -> list[str]:
    """ASE QE-writer species labels: ``Symbol``, ``Symbol1``, ``Symbol2``…

    ASE labels the first occurrence of a chemical symbol with the bare
    symbol; subsequent occurrences (driven by magmom splitting) get a
    numeric suffix. The HUBBARD card references species by these
    labels — getting them wrong silently puts U on the wrong sublattice.
    """
    counts: dict[str, int] = {}
    labels: list[str] = []
    for _, symbol in species_pairs:
        n = counts.get(symbol, 0)
        labels.append(symbol if n == 0 else f"{symbol}{n}")
        counts[symbol] = n + 1
    return labels


def _build_hubbard_card(
    species_pairs: Sequence[tuple[float, str]],
    species_labels: Sequence[str],
    hubbard_u: Mapping[str, float],
    manifolds: Mapping[str, str],
    projector_type: str,
) -> str:
    """Emit the QE 7.1+ ``HUBBARD {projector}\\nU label-manifold value`` block.

    Returns an empty string if no species in ``species_pairs`` has a U
    value configured — saves the caller from having to special-case
    "no Hubbard" runs.
    """
    relevant: list[tuple[str, str, float]] = []
    for (_, symbol), label in zip(species_pairs, species_labels, strict=True):
        if symbol not in hubbard_u:
            continue
        if symbol not in manifolds:
            raise KeyError(
                f"No Hubbard manifold registered for symbol {symbol!r}; "
                f"add it to `manifolds=` or to DEFAULT_HUBBARD_MANIFOLDS."
            )
        relevant.append((label, manifolds[symbol], float(hubbard_u[symbol])))

    if not relevant:
        return ""

    # Documented QE 7.x syntax is `HUBBARD { projector_type }` with spaces
    # inside the braces. The Fortran parser is whitespace-tolerant, but match
    # the docs exactly so we never have to debate it again.
    lines = [f"HUBBARD {{ {projector_type} }}"]
    for label, manifold, u_value in relevant:
        # QE accepts plain decimal; six digits is enough for any U we'd quote.
        lines.append(f"U {label}-{manifold} {u_value:.6f}")
    return "\n".join(lines) + "\n"


def spin_and_hubbard_overrides(
    atoms: Atoms,
    *,
    nspin: int = 2,
    hubbard_u: Mapping[str, float] | None = None,
    starting_magnetization: Mapping[str, float] | Sequence[float] | None = None,
    projector_type: str = DEFAULT_HUBBARD_PROJECTOR_TYPE,
    manifolds: Mapping[str, str] | None = None,
) -> SpinHubbardOverrides:
    """Build the spin + Hubbard-U pieces for a QE 7.1+ pw.x input.

    QE 7.1 split Hubbard U out of the ``&SYSTEM`` namelist into a
    dedicated ``HUBBARD`` card; older inputs hit
    ``DFT+Hubbard input syntax has changed since v7.1`` and abort.
    This helper returns both pieces in one shot:

    * The ``&SYSTEM`` namelist gets ``nspin`` and any
      ``starting_magnetization(i)`` entries (unchanged from pre-7.1).
    * The HUBBARD card text — emitted only when ``hubbard_u`` is given —
      lists one ``U <label>-<manifold> <value>`` line per species,
      including the AFM-split duplicates (e.g. CuO's ``Cu`` + ``Cu1``).

    Species index / label assignment mirrors ASE's QE writer. With
    ``nspin=2`` (or any nonzero initial magmom on ``atoms``), ASE
    splits atoms of the same chemical symbol into separate species
    when their initial magnetic moments differ — e.g. CuO's AFM
    ordering produces species ``Cu`` (mag +1) and ``Cu1`` (mag -1),
    both still Cu chemically, both needing the same Hubbard U.

    For per-atom AFM starting moments (alternating +/- on Cu in CuO),
    set ``atoms.set_initial_magnetic_moments(...)`` BEFORE calling
    :func:`write_pw_input`; ASE will write per-species
    ``starting_magnetization(i)`` cards derived from those values and
    will overwrite any ``starting_magnetization`` you pass through
    this helper. The :func:`build_bulk_cuo` builder sets the AFM
    moments automatically.

    Args:
        atoms: Structure being computed. Used to determine the species
            list (chemical symbols + per-atom magnetic moments).
        nspin: 1 (non-magnetic) or 2 (collinear spin-polarized).
        hubbard_u: Mapping from chemical symbol to U value in eV.
            Species not present in the structure are silently ignored.
        starting_magnetization: Mapping from symbol to initial moment,
            OR a sequence indexed by species order. Use this only
            when ``atoms`` has no per-atom magmoms set (otherwise ASE
            overrides). For AFM, set per-atom magmoms on ``atoms``.
        projector_type: QE HUBBARD-card projector. Default
            :data:`DEFAULT_HUBBARD_PROJECTOR_TYPE` (``"atomic"``)
            reproduces the pre-7.1 ``lda_plus_u_kind=0`` semantics so
            literature U values stay comparable.
        manifolds: Override / extend the default
            :data:`DEFAULT_HUBBARD_MANIFOLDS` (``{"Cu": "3d", ...}``).
            Unknown symbols raise ``KeyError`` rather than silently
            dropping the U term.

    Returns:
        :class:`SpinHubbardOverrides` carrying the namelist overrides
        and (optionally) the HUBBARD card text.

    Example:
        >>> from copper_oxide_dft.structure_builder import (
        ...     build_bulk_cu2o, build_bulk_cuo
        ... )
        >>> # Non-magnetic Cu2O: one Cu species, U on Cu.
        >>> ov = spin_and_hubbard_overrides(
        ...     build_bulk_cu2o(), nspin=1, hubbard_u={"Cu": 4.0}
        ... )
        >>> ov.namelist_overrides["system"]["nspin"]
        1
        >>> "U Cu-3d 4.000000" in ov.hubbard_card
        True
        >>> # AFM CuO: two Cu species (the +/- sublattices), U on BOTH.
        >>> ov = spin_and_hubbard_overrides(
        ...     build_bulk_cuo(), nspin=2, hubbard_u={"Cu": 4.0}
        ... )
        >>> "U Cu-3d 4.000000" in ov.hubbard_card
        True
        >>> "U Cu1-3d 4.000000" in ov.hubbard_card
        True
    """
    if nspin not in (1, 2):
        raise ValueError(f"nspin must be 1 or 2, got {nspin}")

    species_pairs = _ase_species_breakdown(atoms, nspin=nspin)

    system: dict[str, Any] = {"nspin": nspin}

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

    hubbard_card = ""
    if hubbard_u:
        species_labels = _ase_species_labels(species_pairs)
        manifold_map = dict(DEFAULT_HUBBARD_MANIFOLDS)
        if manifolds:
            manifold_map.update(manifolds)
        hubbard_card = _build_hubbard_card(
            species_pairs,
            species_labels,
            hubbard_u,
            manifold_map,
            projector_type,
        )

    return SpinHubbardOverrides(
        namelist_overrides={"system": system},
        hubbard_card=hubbard_card,
    )


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
