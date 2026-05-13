"""Atomic structure builders for the copper-oxide-dft project.

Wraps ASE constructors with project-specific conventions. Phase 0 / Phase 1
provides only bulk Cu; later phases add slabs, oxide bulks, overlayers, and
adsorbates.
"""

from __future__ import annotations

from ase import Atoms
from ase.build import bulk

CU_LATTICE_PARAMETER_ANG = 3.615
"""Experimental fcc Cu lattice parameter (Å). Used as the starting point for
variable-cell relaxation; the relaxed value should land within ~0.5 % of this."""


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
