"""Atomic structure builders and inspection helpers.

Wraps ASE constructors with project-specific conventions. Phase 0 / Phase 1
provides only bulk Cu; later phases add slabs, oxide bulks, overlayers, and
adsorbates.

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

from ase import Atoms
from ase.build import bulk

CU_LATTICE_PARAMETER_ANG = 3.615
"""Experimental fcc Cu lattice parameter (Å). Used as the starting point for
variable-cell relaxation; the relaxed value should land within ~0.5 % of this."""


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
