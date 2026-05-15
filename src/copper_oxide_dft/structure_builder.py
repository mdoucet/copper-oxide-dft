"""Atomic structure builders and inspection helpers.

Wraps ASE constructors with project-specific conventions. Phase 1 covers bulk
Cu; Phase 2 adds the bulk oxides Cu2O (cuprite, non-magnetic) and CuO (tenorite,
antiferromagnetic). Reference molecules H2O / H2 / O2 are also provided here —
they're needed as chemical-potential references for the Phase 4 Computational
Hydrogen Electrode (CHE) post-processing, and live with the other structure
builders for discoverability.

:func:`summarize_layers` groups atoms by z-coordinate into layers, so a
slab (or any extended structure) can be visually validated before
submitting expensive calculations: do the atoms stack in the right
order, are the layer spacings sensible, did the surface terminate
correctly?
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from ase import Atom, Atoms
from ase.build import add_adsorbate, bulk, fcc111, surface
from ase.constraints import FixAtoms

CU_LATTICE_PARAMETER_ANG = 3.615
"""Experimental fcc Cu lattice parameter (Å). Used as the starting point for
variable-cell relaxation; the relaxed value should land within ~0.5 % of this."""

CU2O_LATTICE_PARAMETER_ANG = 4.2696
"""Experimental Cu2O cuprite lattice parameter (Å), cubic Pn-3m, a=b=c.
Restrepo et al., AIP Adv. 4, 027119 (2014). Cu atoms occupy the 4b Wyckoff
positions (fcc sublattice); O occupies the 2a positions (bcc sublattice).
The conventional cell is 6 atoms (4 Cu + 2 O); Cu2O is non-magnetic."""

CUO_LATTICE_PARAMETERS_ANG = (4.6837, 3.4226, 5.1288)
"""Experimental CuO tenorite lattice parameters (a, b, c) in Å, monoclinic
C2/c (#15). Asbrink & Norrby, Acta Cryst. B26, 8 (1970)."""

CUO_BETA_DEG = 99.54
"""Monoclinic angle for CuO tenorite (degrees). Same reference as
:data:`CUO_LATTICE_PARAMETERS_ANG`."""

DEFAULT_SLAB_LAYERS = 4
"""Default Cu(111) slab thickness. 4 layers is the project standard
(see implementation-plan.md): thick enough for converged surface energy,
thin enough to be affordable. Bottom 2 are conventionally fixed to bulk."""

DEFAULT_SLAB_VACUUM_ANG = 15.0
"""Default vacuum thickness above the slab (Å). 15 Å suffices for Cu(111)
in vacuum with dipole correction enabled; bump to 20+ for explicit water."""

DEFAULT_FIXED_BOTTOM_LAYERS = 2
"""How many bottom layers to constrain to bulk geometry during a relaxation.
The point is to mimic semi-infinite bulk underneath the surface: top layers
relax freely, bottom layers don't."""


@dataclass(frozen=True)
class Layer:
    """One z-layer of an ASE structure.

    A layer is the set of atoms whose z-coordinates fall inside the
    same tolerance window. ``z`` is the mean z of the layer; ``elements``
    counts species in the layer; ``thickness`` is the (max-min) spread.
    """

    z: float
    elements: Mapping[str, int]
    thickness: float

    @property
    def total_atoms(self) -> int:
        return sum(self.elements.values())

    def composition_label(self) -> str:
        return " ".join(f"{el}x{n}" for el, n in sorted(self.elements.items()))


def build_bulk_cu(a: float = CU_LATTICE_PARAMETER_ANG) -> Atoms:
    """Build a primitive-cell fcc Cu bulk structure.

    Args:
        a: Conventional cubic lattice parameter in Å.

    Returns:
        ASE Atoms with a 1-atom primitive fcc cell.

    Example:
        >>> atoms = build_bulk_cu()
        >>> atoms.get_chemical_formula()
        'Cu'
        >>> len(atoms)
        1
    """
    return bulk("Cu", crystalstructure="fcc", a=a)


def build_bulk_cu2o(a: float = CU2O_LATTICE_PARAMETER_ANG) -> Atoms:
    """Build the conventional cubic Cu2O (cuprite) bulk cell.

    Cu2O is non-magnetic and has a measured band gap of 2.17 eV, which
    PBE underestimates (~0.5 eV) but PBE+U corrects toward experiment.
    Space group Pn-3m (#224): Cu at 4b sites (1/4, 1/4, 1/4 etc.), O at
    2a sites (0, 0, 0 and 1/2, 1/2, 1/2). 6 atoms in the conventional
    cell.

    Args:
        a: Cubic lattice parameter in Å. Default is the experimental
            value (4.27 Å). Pass relaxed value when running production
            calculations on top of a vc-relax result.

    Returns:
        ASE Atoms with 4 Cu + 2 O in a cubic cell.

    Example:
        >>> atoms = build_bulk_cu2o()
        >>> sorted(atoms.get_chemical_symbols())
        ['Cu', 'Cu', 'Cu', 'Cu', 'O', 'O']
        >>> len(atoms)
        6
    """
    # Pn-3m Wyckoff positions for Cu2O (Cu = 4b, O = 2a) in fractional coords.
    scaled_positions = [
        (0.25, 0.25, 0.25),
        (0.25, 0.75, 0.75),
        (0.75, 0.25, 0.75),
        (0.75, 0.75, 0.25),
        (0.0, 0.0, 0.0),
        (0.5, 0.5, 0.5),
    ]
    return Atoms(
        symbols=["Cu", "Cu", "Cu", "Cu", "O", "O"],
        scaled_positions=scaled_positions,
        cell=[a, a, a],
        pbc=True,
    )


def build_bulk_cuo(
    a: float = CUO_LATTICE_PARAMETERS_ANG[0],
    b: float = CUO_LATTICE_PARAMETERS_ANG[1],
    c: float = CUO_LATTICE_PARAMETERS_ANG[2],
    beta_deg: float = CUO_BETA_DEG,
) -> Atoms:
    """Build the conventional monoclinic CuO (tenorite) bulk cell.

    CuO is antiferromagnetic; ``nspin=2`` with explicit per-atom starting
    magnetizations is mandatory or the calculation falls into a wrong
    (non-magnetic or ferromagnetic) local minimum. This builder assigns
    +1 / -1 starting moments to alternating Cu atoms along the c axis,
    which matches the experimental AFM-II ordering. Override via
    :func:`set_cuo_magnetic_ordering` if a different ordering is needed.

    Space group C2/c (#15) with Cu at 4c (0.25, 0.25, 0) and O at 4e
    (0, y, 0.25) with y ≈ 0.4184. 8 atoms in the conventional cell
    (4 Cu + 4 O).

    Args:
        a, b, c: Monoclinic lattice parameters in Å (defaults are the
            experimental values).
        beta_deg: Monoclinic angle in degrees (default 99.54°).

    Returns:
        ASE Atoms with 4 Cu + 4 O and ``initial_magnetic_moments`` set
        to the AFM-II ordering.

    Example:
        >>> atoms = build_bulk_cuo()
        >>> sorted(atoms.get_chemical_symbols()).count('Cu')
        4
        >>> sorted(atoms.get_chemical_symbols()).count('O')
        4
    """
    y_o = 0.4184
    # C2/c (#15) generators on Cu (4c) and O (4e) sites.
    scaled_positions = [
        (0.25, 0.25, 0.0),
        (0.75, 0.75, 0.0),
        (0.25, 0.75, 0.5),
        (0.75, 0.25, 0.5),
        (0.0, y_o, 0.25),
        (0.0, -y_o, 0.75),
        (0.5, 0.5 + y_o, 0.25),
        (0.5, 0.5 - y_o, 0.75),
    ]
    # Monoclinic cell with b unique (standard QE convention): a along x,
    # b along y, c in the x-z plane tilted by beta from a.
    beta = np.deg2rad(beta_deg)
    cell = np.array(
        [
            [a, 0.0, 0.0],
            [0.0, b, 0.0],
            [c * np.cos(beta), 0.0, c * np.sin(beta)],
        ]
    )
    atoms = Atoms(
        symbols=["Cu", "Cu", "Cu", "Cu", "O", "O", "O", "O"],
        scaled_positions=scaled_positions,
        cell=cell,
        pbc=True,
    )
    # AFM-II ordering: pairs of Cu along c get +/- starting moments.
    # Indices 0,1 are at z=0 (set +1, -1); indices 2,3 at z=c/2 (-1, +1).
    atoms.set_initial_magnetic_moments([+1.0, -1.0, -1.0, +1.0, 0, 0, 0, 0])
    return atoms


def build_cu111_slab(
    *,
    layers: int = DEFAULT_SLAB_LAYERS,
    supercell: tuple[int, int] = (3, 3),
    vacuum_ang: float = DEFAULT_SLAB_VACUUM_ANG,
    a: float = CU_LATTICE_PARAMETER_ANG,
    fix_bottom_layers: int = DEFAULT_FIXED_BOTTOM_LAYERS,
) -> Atoms:
    """Build a Cu(111) slab with project conventions.

    Cu(111) is the close-packed face of fcc copper and the experimentally
    relevant surface for most Cu electrochemistry. ASE's :func:`fcc111`
    builds the primitive (1×1) cell with a chosen layer count and
    vacuum; we expand it to ``supercell`` lateral repeats and apply the
    project-standard bottom-layer constraint.

    The bottom ``fix_bottom_layers`` layers are tagged with a
    :class:`ase.constraints.FixAtoms` constraint so a relax/vc-relax
    doesn't move them — they stand in for the semi-infinite bulk that
    a real surface would have underneath.

    Args:
        layers: Number of (111) layers in the slab.
        supercell: Lateral repetition ``(nx, ny)`` of the primitive cell.
            A 3×3 cell gives 1/9 ML adsorbate coverage resolution; use 2×2
            for cheap tests or 4×4 for finer coverage steps.
        vacuum_ang: Vacuum thickness above the slab (Å). With dipole
            correction, 15 Å is enough; without, bump to 20+.
        a: fcc Cu lattice parameter (Å). Use the Phase 1 converged value
            in production.
        fix_bottom_layers: Number of bottom layers constrained to bulk
            positions during relaxation.

    Returns:
        ASE :class:`Atoms` with the slab geometry, a vacuum gap, and a
        ``FixAtoms`` constraint on the bottom layers. The slab is built
        in the standard "bottom at low z" orientation.

    Raises:
        ValueError: If ``fix_bottom_layers`` exceeds ``layers``.

    Example:
        >>> slab = build_cu111_slab(layers=4, supercell=(3, 3))
        >>> # 4 layers x 9 atoms per layer = 36 atoms.
        >>> len(slab)
        36
    """
    if fix_bottom_layers > layers:
        raise ValueError(
            f"Cannot fix {fix_bottom_layers} layers in a {layers}-layer slab."
        )

    slab = fcc111("Cu", size=(supercell[0], supercell[1], layers), a=a, vacuum=vacuum_ang)
    if fix_bottom_layers > 0:
        slab.set_constraint(_fix_bottom_layers(slab, fix_bottom_layers))
    return slab


def add_oxygen_adsorbates(
    slab: Atoms,
    *,
    coverage_ml: float,
    site: str = "fcc",
    height_ang: float = 1.5,
    adsorbate: str = "O",
) -> Atoms:
    """Place O (or OH) adsorbates on a Cu(111) slab at a fractional coverage.

    Coverage is expressed in monolayers (ML), where 1 ML means one
    adsorbate per surface metal atom. For a 3×3 cell, the achievable
    coverages on top of the surface layer are 1/9, 2/9, 1/3, … up to 1
    ML. The function picks ``round(coverage_ml * n_surface)`` sites of
    the requested type from the top layer; if the requested coverage
    rounds to zero, raises an error (so a "1/9 ML" request on a 2×2 cell
    fails loudly rather than silently dropping the adsorbate).

    Args:
        slab: Cu(111) slab from :func:`build_cu111_slab` (or compatible).
        coverage_ml: Target coverage in monolayers (0 < c ≤ 1).
        site: One of ``"top"``, ``"bridge"``, ``"fcc"`` (3-fold hollow over
            a hcp atom of the next layer), ``"hcp"`` (3-fold hollow over a
            second-layer atom). FCC-hollow is the canonical lowest-energy
            site for O on Cu(111); use ``top`` for diagnostic comparisons.
        height_ang: Distance above the top-layer z (Å). 1.5 Å is a
            reasonable starting guess for O on Cu(111); the relaxation
            will adjust to ~1.3–1.4 Å.
        adsorbate: ``"O"`` or ``"OH"`` (or any short chemical symbol that
            ASE's :func:`add_adsorbate` understands). For ``"OH"`` we
            place the O on the surface and the H 0.96 Å above it.

    Returns:
        A new :class:`Atoms` (the input is copied) with adsorbates added
        on top of the original slab. Any existing constraint on the
        slab is preserved.

    Raises:
        ValueError: If ``coverage_ml`` is out of (0, 1] or
            ``round(coverage_ml * n_surface)`` is zero.

    Example:
        >>> slab = build_cu111_slab(supercell=(3, 3))
        >>> covered = add_oxygen_adsorbates(slab, coverage_ml=1/9, site="fcc")
        >>> from collections import Counter
        >>> Counter(covered.get_chemical_symbols())["O"]
        1
    """
    if not 0.0 < coverage_ml <= 1.0:
        raise ValueError(
            f"coverage_ml must be in (0, 1]; got {coverage_ml}."
        )
    if site not in {"top", "bridge", "fcc", "hcp"}:
        raise ValueError(
            f"Unknown site {site!r}; expected one of top, bridge, fcc, hcp."
        )

    top_layer = summarize_layers(slab)[-1]
    n_surface_atoms = top_layer.total_atoms
    n_to_place = round(coverage_ml * n_surface_atoms)
    if n_to_place == 0:
        raise ValueError(
            f"coverage_ml={coverage_ml} on a {n_surface_atoms}-atom surface "
            "rounds to zero adsorbates. Use a larger supercell or coverage."
        )

    surface_indices = [
        i for i, atom in enumerate(slab) if abs(atom.z - top_layer.z) < 0.5
    ]
    if n_to_place > len(surface_indices):
        raise ValueError(
            f"Requested {n_to_place} sites but only {len(surface_indices)} "
            "available surface atoms; cap coverage at 1 ML."
        )
    chosen = sorted(surface_indices)[:n_to_place]

    result = slab.copy()
    for idx in chosen:
        # ASE's add_adsorbate accepts a 2D position; we use the surface
        # atom's (x, y) shifted into the requested high-symmetry site.
        add_adsorbate(
            result,
            _adsorbate_molecule(adsorbate),
            height=height_ang,
            position=_site_offset(slab, idx, site),
        )
    return result


def _adsorbate_molecule(name: str) -> Atoms:
    if name == "O":
        return Atoms("O", positions=[(0.0, 0.0, 0.0)])
    if name == "OH":
        # H sits 0.96 Å above O, vertical; not the lowest-energy OH
        # orientation but a reasonable starting guess for relaxation.
        return Atoms("OH", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.96)])
    # Single-atom fallback: trust the caller knows what they're doing.
    return Atoms(name, positions=[(0.0, 0.0, 0.0)])


def _site_offset(slab: Atoms, surface_idx: int, site: str) -> tuple[float, float]:
    """Return the (x, y) of the requested high-symmetry site near a surface atom.

    ``top``: directly above the surface atom.
    ``bridge``: midway between the surface atom and its nearest in-plane
        neighbor along +x of the cell (approximation; fine for starting
        geometry — the relaxation finds the real symmetry-equivalent site).
    ``fcc``: 3-fold hollow ~(0.5 a_eff, 0.29 a_eff) shifted from the surface
        atom, where a_eff is the nearest-neighbor distance.
    ``hcp``: as fcc but mirror-reflected; differs in which sublayer atom
        sits underneath.
    """
    a_nn = float(np.linalg.norm(slab.cell[0]) / max(1, _supercell_repeats_along_x(slab)))
    x, y = float(slab[surface_idx].x), float(slab[surface_idx].y)
    if site == "top":
        return (x, y)
    if site == "bridge":
        return (x + 0.5 * a_nn, y)
    if site == "fcc":
        return (x + 0.5 * a_nn, y + a_nn / (2.0 * np.sqrt(3.0)))
    # hcp
    return (x + 0.5 * a_nn, y - a_nn / (2.0 * np.sqrt(3.0)))


def _supercell_repeats_along_x(slab: Atoms) -> int:
    """Best-effort estimate of how many primitive cells fit along x.

    Used to convert the slab's cell length into a nearest-neighbor
    distance for site offsets. Falls back to 1 for unusual slabs.
    """
    atoms_per_layer = max(layer.total_atoms for layer in summarize_layers(slab))
    return max(1, int(np.sqrt(atoms_per_layer)))


def surface_energy_ev_per_a2(
    slab_energy_ev: float,
    bulk_energy_per_atom_ev: float,
    n_atoms_in_slab: int,
    surface_area_ang2: float,
    *,
    n_surfaces: int = 2,
) -> float:
    """Surface energy from total energies of a slab and the underlying bulk.

    ``γ = (E_slab − N · E_bulk_per_atom) / (n_surfaces · A)``

    Standard formula for a stoichiometric slab. ``n_surfaces`` is 2 for a
    symmetric (vacuum on both sides) slab and 1 for an asymmetric slab
    where dipole correction handles one side.

    Args:
        slab_energy_ev: Total DFT energy of the relaxed slab (eV).
        bulk_energy_per_atom_ev: DFT energy of bulk Cu per atom (eV)
            (use ``parse_pw_output`` on the bulk vc-relax run).
        n_atoms_in_slab: Number of atoms in the slab.
        surface_area_ang2: In-plane area of the slab cell (Å²).
        n_surfaces: Number of surfaces exposed to vacuum (2 for symmetric
            slab; 1 if a dipole correction effectively suppresses one
            side).

    Returns:
        Surface energy in eV/Å². Cu(111) literature value is ~0.08
        eV/Å² (≈1.3 J/m²).
    """
    if surface_area_ang2 <= 0:
        raise ValueError(
            f"surface_area_ang2 must be positive; got {surface_area_ang2}."
        )
    if n_surfaces not in (1, 2):
        raise ValueError(f"n_surfaces must be 1 or 2; got {n_surfaces}.")
    cleave_energy = slab_energy_ev - n_atoms_in_slab * bulk_energy_per_atom_ev
    return cleave_energy / (n_surfaces * surface_area_ang2)


def add_explicit_water_layer(
    slab: Atoms,
    *,
    n_waters: int,
    height_ang: float = 2.5,
    layer_thickness_ang: float = 3.0,
    seed: int = 0,
) -> Atoms:
    """Add a layer of explicit H2O molecules above a slab.

    Phase 6 scaffold: places ``n_waters`` water molecules above the
    topmost slab z, spread on a 2D grid covering the lateral cell.
    Orientation is randomized (deterministic for a given ``seed``) so
    that subsequent MD/relaxation has a sensible starting point. This
    is a *starting guess* only; users should pre-equilibrate the
    water layer via classical MD or short AIMD before the production
    DFT run.

    Args:
        slab: Underlying slab (typically Cu(111) with optional adsorbates).
        n_waters: Number of H2O molecules to add.
        height_ang: Distance from the topmost slab atom to the bottom
            of the water layer (Å).
        layer_thickness_ang: Vertical extent of the water layer (Å).
            Waters are distributed uniformly across this slab in z.
        seed: RNG seed for reproducible orientations.

    Returns:
        New :class:`Atoms` with the water layer added.

    Raises:
        ValueError: If ``n_waters`` is negative.
    """
    if n_waters < 0:
        raise ValueError(f"n_waters must be non-negative; got {n_waters}.")

    result = slab.copy()
    if n_waters == 0:
        return result

    top_layer = summarize_layers(slab)[-1]
    z0 = top_layer.z + height_ang

    a_vec = np.asarray(slab.cell[0])
    b_vec = np.asarray(slab.cell[1])

    rng = np.random.default_rng(seed)

    # Distribute waters on a near-square grid covering the (a, b) plane.
    n_x = max(1, int(np.ceil(np.sqrt(n_waters))))
    n_y = max(1, int(np.ceil(n_waters / n_x)))

    bond = 0.9572
    angle = np.deg2rad(104.5)
    placed = 0
    for ix in range(n_x):
        for iy in range(n_y):
            if placed >= n_waters:
                break
            frac_x = (ix + 0.5) / n_x
            frac_y = (iy + 0.5) / n_y
            origin = frac_x * a_vec + frac_y * b_vec
            # Spread waters across the layer thickness uniformly in z.
            z_offset = layer_thickness_ang * placed / max(1, n_waters - 1) if n_waters > 1 else 0.0
            o_pos = (origin[0], origin[1], z0 + z_offset)
            # Random rotation around z so adjacent waters don't all align.
            theta = float(rng.uniform(0, 2 * np.pi))
            h_x = bond * np.sin(angle / 2.0)
            h_z = bond * np.cos(angle / 2.0)
            h1 = (
                o_pos[0] + h_x * np.cos(theta),
                o_pos[1] + h_x * np.sin(theta),
                o_pos[2] + h_z,
            )
            h2 = (
                o_pos[0] - h_x * np.cos(theta),
                o_pos[1] - h_x * np.sin(theta),
                o_pos[2] + h_z,
            )
            result.append(Atom("O", position=o_pos))
            result.append(Atom("H", position=h1))
            result.append(Atom("H", position=h2))
            placed += 1
    return result


def build_cu2o_111_slab(
    *,
    layers: int = 3,
    supercell: tuple[int, int] = (1, 1),
    vacuum_ang: float = DEFAULT_SLAB_VACUUM_ANG,
    a: float = CU2O_LATTICE_PARAMETER_ANG,
    fix_bottom_layers: int = 1,
) -> Atoms:
    """Build a Cu2O(111) slab via ASE's general ``surface`` builder.

    This is the lowest-resolution sensible builder: the (111) termination
    of cuprite has multiple non-equivalent terminations (Cu-, O-,
    Cu2O-trilayer), and which is lowest in energy is a Phase 3 finding,
    not a baked-in default. The builder produces *a* (111) slab from
    the cuprite bulk; verify termination via ``inspect`` and re-cut at
    a different z if needed.

    Args:
        layers: Number of Cu2O trilayers (Cu-O-Cu).
        supercell: Lateral (nx, ny) repeats of the (1×1) Cu2O(111) cell.
        vacuum_ang: Vacuum thickness (Å).
        a: Cu2O cubic lattice parameter (Å); use the relaxed Phase 2
            value in production.
        fix_bottom_layers: Number of bottom z-layers to constrain to bulk.

    Returns:
        ASE :class:`Atoms` with the slab and a FixAtoms constraint.
    """
    bulk_cell = build_bulk_cu2o(a=a)
    slab = surface(bulk_cell, indices=(1, 1, 1), layers=layers, vacuum=vacuum_ang)
    slab *= (supercell[0], supercell[1], 1)
    if fix_bottom_layers > 0:
        slab.set_constraint(_fix_bottom_layers(slab, fix_bottom_layers))
    return slab


def build_cuo_111_slab(
    *,
    layers: int = 3,
    supercell: tuple[int, int] = (1, 1),
    vacuum_ang: float = DEFAULT_SLAB_VACUUM_ANG,
    fix_bottom_layers: int = 1,
) -> Atoms:
    """Build a CuO(111) slab via ASE's general ``surface`` builder.

    Tenorite is monoclinic, so (111) is unambiguous but produces a
    slab whose termination depends on where the cleave plane intersects
    the unit cell. Like the Cu2O(111) builder, this returns *a*
    termination; verify via ``inspect``. AFM starting moments are
    propagated from the bulk to the slab.

    Args:
        layers: Number of (111) layers.
        supercell: Lateral (nx, ny) repeats.
        vacuum_ang: Vacuum thickness (Å).
        fix_bottom_layers: Number of bottom z-layers to constrain.

    Returns:
        ASE :class:`Atoms` with the slab, AFM moments on Cu, and a
        FixAtoms constraint.
    """
    bulk_cell = build_bulk_cuo()
    slab = surface(bulk_cell, indices=(1, 1, 1), layers=layers, vacuum=vacuum_ang)
    slab *= (supercell[0], supercell[1], 1)
    if fix_bottom_layers > 0:
        slab.set_constraint(_fix_bottom_layers(slab, fix_bottom_layers))
    return slab


def _fix_bottom_layers(slab: Atoms, n_layers: int) -> FixAtoms:
    """FixAtoms constraint that pins atoms in the bottom ``n_layers`` z-layers.

    Uses :func:`summarize_layers` to identify layers; mirrors the layer
    tolerance the inspect command uses so what you see in `inspect` is
    what's constrained.
    """
    layers = summarize_layers(slab)
    if n_layers >= len(layers):
        # Pin everything (degenerate but valid for sanity tests).
        return FixAtoms(indices=list(range(len(slab))))
    threshold_z = layers[n_layers - 1].z + 1e-3
    indices = [i for i, atom in enumerate(slab) if atom.z <= threshold_z]
    return FixAtoms(indices=indices)


def build_reference_h2o(box_size_ang: float = 12.0) -> Atoms:
    """Isolated H2O molecule in a cubic box for QE reference-energy runs.

    The CHE post-processing needs μ(H2O), μ(H2), and μ(O2) computed at the
    same level of theory as the bulk/slab calculations. Each reference is
    a single molecule in a large enough box that periodic images do not
    interact (12 Å is the standard safe choice for Q-E with the project's
    cutoffs; bump higher if a convergence test demands it).

    Args:
        box_size_ang: Edge length of the cubic simulation cell (Å).

    Returns:
        ASE Atoms with one H2O molecule centered in the box.
    """
    # Equilibrium geometry: O-H ~0.96 Å, H-O-H angle ~104.5°.
    half = box_size_ang / 2.0
    bond = 0.9572
    angle = np.deg2rad(104.5)
    h_x = bond * np.sin(angle / 2.0)
    h_z = bond * np.cos(angle / 2.0)
    return Atoms(
        symbols=["O", "H", "H"],
        positions=[
            (half, half, half),
            (half + h_x, half, half + h_z),
            (half - h_x, half, half + h_z),
        ],
        cell=[box_size_ang, box_size_ang, box_size_ang],
        pbc=True,
    )


def build_reference_h2(box_size_ang: float = 12.0) -> Atoms:
    """Isolated H2 molecule in a cubic box for QE reference-energy runs.

    Used to compute μ(H2) for the Computational Hydrogen Electrode: the
    CHE replaces μ(H+ + e-) with (1/2)μ(H2) at U=0, pH=0.

    Args:
        box_size_ang: Edge length of the cubic simulation cell (Å).

    Returns:
        ASE Atoms with one H2 molecule centered in the box.
    """
    half = box_size_ang / 2.0
    bond = 0.7414
    return Atoms(
        symbols=["H", "H"],
        positions=[(half - bond / 2.0, half, half), (half + bond / 2.0, half, half)],
        cell=[box_size_ang, box_size_ang, box_size_ang],
        pbc=True,
    )


def build_reference_o2(box_size_ang: float = 12.0) -> Atoms:
    """Isolated O2 molecule (triplet ground state) in a cubic box.

    O2 is paramagnetic with a triplet ground state (S=1, two unpaired
    electrons). The starting magnetic moments are set so that QE finds
    the correct triplet rather than a singlet; ``nspin=2`` is required.

    Args:
        box_size_ang: Edge length of the cubic simulation cell (Å).

    Returns:
        ASE Atoms with one O2 molecule centered in the box and
        ``initial_magnetic_moments`` set to (+1, +1) for the triplet.
    """
    half = box_size_ang / 2.0
    bond = 1.2075
    atoms = Atoms(
        symbols=["O", "O"],
        positions=[(half - bond / 2.0, half, half), (half + bond / 2.0, half, half)],
        cell=[box_size_ang, box_size_ang, box_size_ang],
        pbc=True,
    )
    atoms.set_initial_magnetic_moments([1.0, 1.0])
    return atoms


def summarize_layers(atoms: Atoms, tol: float = 0.1) -> list[Layer]:
    """Group atoms by z-coordinate into layers, ordered bottom to top.

    Two atoms are in the same layer if their z-coordinates differ by at
    most ``tol`` (Å). For a metallic slab the natural value is ~0.1 Å;
    for a strongly relaxed surface use a larger tolerance to keep
    chemically equivalent atoms together.

    Args:
        atoms: ASE structure to summarize.
        tol: z-grouping tolerance (Å).

    Returns:
        Layers sorted by mean z, ascending. Empty list if ``atoms`` is empty.
    """
    if len(atoms) == 0:
        return []
    if tol < 0:
        raise ValueError(f"tol must be non-negative, got {tol}")

    zs = atoms.positions[:, 2]
    order = sorted(range(len(atoms)), key=lambda i: zs[i])

    layers: list[Layer] = []
    bucket_zs: list[float] = []
    bucket_elements: Counter[str] = Counter()

    def flush() -> None:
        if not bucket_zs:
            return
        layers.append(
            Layer(
                z=sum(bucket_zs) / len(bucket_zs),
                elements=dict(bucket_elements),
                thickness=max(bucket_zs) - min(bucket_zs),
            )
        )

    for i in order:
        z = float(zs[i])
        sym = atoms[i].symbol
        if bucket_zs and z - bucket_zs[-1] > tol:
            flush()
            bucket_zs = []
            bucket_elements = Counter()
        bucket_zs.append(z)
        bucket_elements[sym] += 1
    flush()
    return layers
