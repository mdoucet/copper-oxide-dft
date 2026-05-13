# Project Overview

## Project Name

**copper-oxide-dft** — Python pipeline for DFT calculations of copper oxide phases on copper surfaces under electrochemical conditions.

## Purpose

Predict the structure and stability of copper oxide phases (Cu, Cu₂O, CuO, adsorbed O/OH) on Cu(111) under applied electrochemical potential (−1 V to +1 V vs. SHE) in aqueous solution at controlled pH, using Quantum ESPRESSO. The end goal is to identify potential‑driven surface reconstructions; intermediate milestones produce Pourbaix‑style stability diagrams that can be validated against experiment.

## Target Users

Primarily the project owner (computational scientist at ORNL, new to DFT). Designed so the workflow is reproducible by other group members and the analysis steps are accessible to experimentalists.

## Core Functionality

1. **Structure generation** — build bulk, slab, and oxide/Cu interface structures with ASE (Cu(111), Cu₂O(111), CuO(111), adsorbed O/OH at varying coverage).
2. **QE input generation** — generate `pw.x` input files with project‑wide defaults (PBE+U, Marzari–Vanderbilt smearing, PseudoDojo pseudopotentials, dipole correction).
3. **HPC job submission** — emit SLURM scripts for ORNL Andes (CPU) and Frontier (GPU).
4. **Output parsing** — extract energies, forces, magnetizations, Fermi levels from QE output.
5. **Electrochemistry post‑processing** — apply the Computational Hydrogen Electrode (CHE) correction and build Pourbaix‑like (U, pH) stability diagrams.
6. **(Later phases)** — interface to Environ (implicit solvation), ESM‑FCP (constant‑potential DFT), and AIMD trajectories for surface reconstruction studies.

## Input/Output

**Input:**
- High‑level configuration (system type, surface facet, coverage, potential range, pH).
- Pseudopotential files (PseudoDojo PBE).
- HPC cluster identity (Andes/Frontier) for submission script tailoring.

**Output:**
- QE input/output files in a structured directory tree per calculation.
- Parsed results (CSV/JSON): energies, magnetizations, geometries.
- Plots: convergence curves, surface energy vs. coverage, Pourbaix diagrams.

## Example Usage

```python
from cuox_dft.structure_builder import build_cu_slab, add_oxide_overlayer
from cuox_dft.qe_input import write_relax_input
from cuox_dft.submit import submit_andes

slab = build_cu_slab(facet=(1, 1, 1), layers=4, supercell=(3, 3, 1), vacuum=15.0)
slab = add_oxide_overlayer(slab, oxide="Cu2O", coverage=0.5)

write_relax_input(slab, out="runs/cu2o_cu111_0.5ML/")
submit_andes("runs/cu2o_cu111_0.5ML/", nodes=4, walltime="6:00:00")
```

```bash
# CLI equivalent
cuox-dft converge --system bulk-cu --param ecutwfc --range 40-100 --step 20
cuox-dft pourbaix --phases Cu,Cu2O,CuO --u-range -1,1 --ph 7
```

## Dependencies

- **Core**: `ase`, `pymatgen`, `numpy`, `matplotlib`
- **CLI**: `click`
- **Testing**: `pytest`, `pytest-cov`
- **External (not pip)**: Quantum ESPRESSO (`pw.x`, `hp.x`, optionally Environ patch and ESM‑RISM); PseudoDojo pseudopotential library; SLURM scheduler on ORNL HPC.

## Technical Notes

- **Functional**: PBE + Hubbard U on Cu 3d (U ≈ 4–7 eV, exact value chosen via literature or `hp.x` linear response).
- **Spin polarization**: required for CuO (antiferromagnetic) and any O‑containing surface; not needed for pure Cu or Cu₂O.
- **k‑point sampling**: Γ‑centered Monkhorst–Pack; converged per system size.
- **Smearing**: Marzari–Vanderbilt cold smearing, σ ≈ 0.02 Ry (mandatory for metallic Cu).
- **Slab convention**: 4 layers, bottom 2 fixed at bulk geometry, 15 Å vacuum, dipole correction for asymmetric slabs.
- **Solvation roadmap**: vacuum → Environ implicit (ε=78.4) → explicit water layer → ESM‑RISM.
- **Potential roadmap**: CHE post‑processing (Nørskov) → ESM‑FCP constant‑potential DFT.
- **Compute target**: start on ORNL Andes (CPU, easier to debug); migrate production runs to Frontier (AMD GPU) once converged.

See [implementation-plan.md](implementation-plan.md) for the phased roadmap and [ground_truths.md](ground_truths.md) for methodology decisions and known gotchas.
