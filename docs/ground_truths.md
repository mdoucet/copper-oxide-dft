# Ground Truths

This file captures key findings, decisions, and verified facts discovered during development. It serves as a persistent knowledge base that AI assistants (GitHub Copilot, Claude Code, etc.) and developers can reference across sessions.

**Why this matters:** AI assistants don't remember previous conversations. By recording important discoveries here, you ensure that context isn't lost between sessions. When the assistant reads this file, it can make better suggestions based on what's already been learned about your project.

## How to Use This File

- **Add entries as you discover important facts** — things like API quirks, configuration requirements, performance constraints, or design decisions.
- **Include the date and context** so future-you (or the assistant) understands why something was noted.
- **Link to relevant code or docs** when helpful.
- Both Copilot and Claude Code are instructed to update this file automatically when they discover key findings during development.

## Findings

### 2026-05-13: Project scope and methodology baseline

Initial methodology decisions for the DFT calculations. See [implementation-plan.md](implementation-plan.md) for the phased roadmap and the rationale behind each choice.

**System under study:** Cu(111) with Cu oxide overlayers (Cu₂O, CuO, adsorbed O/OH) under applied potential (−1 V to +1 V vs. SHE) in aqueous solution. End goal: potential-driven surface reconstruction.

**Baseline methodology:**
- Functional: **PBE + Hubbard U on Cu 3d**. Pure PBE is insufficient for Cu oxide band gaps and energetics.
- Pseudopotentials: **PseudoDojo PBE PAW** for Cu, O, H.
- Smearing: **Marzari–Vanderbilt cold smearing, σ ≈ 0.02 Ry** — mandatory for metallic Cu.
- Slab: **4 layers, bottom 2 fixed, 15 Å vacuum, dipole correction**.
- Solvation roadmap: vacuum → Environ implicit (ε=78.4) → explicit H₂O → ESM-RISM.
- Potential roadmap: CHE post-processing → ESM-FCP constant-potential.
- HPC: develop on **ORNL Andes** (CPU), scale production to **Frontier** (AMD GPU).

### 2026-05-13: Cu oxide DFT gotchas (non-obvious)

- **CuO is antiferromagnetic** — `nspin=2` with explicit starting magnetizations is mandatory; non-magnetic CuO is qualitatively wrong (wrong band gap, wrong lattice).
- **Cu 3d electrons need DFT+U** — PBE underestimates Cu₂O band gap (~0.5 eV vs. 2.17 eV experiment). Typical U on Cu d is 4–7 eV (Mosey/Carter ~4 eV is a common literature pick).
- **Metallic Cu requires smearing** — without it, SCF will not converge or will give wrong forces. Marzari–Vanderbilt is the safe choice; do not use Gaussian for metals.
- **Cu₂O on Cu(111) has ~17% lattice mismatch** — needs a coincident supercell, not a simple ×n superlattice.
- **CHE is post-processing, not DFT** — the Nørskov computational hydrogen electrode shifts free energies *after* the DFT calculation by −eU − k_B T·ln(10)·pH; the underlying DFT is neutral. Good for stability diagrams, does not produce the *structure* at a given potential.
- **Constant-potential ≠ CHE** — to actually study reconstruction at potential, you need ESM-FCP (charged slab + counter-electrode) or ESM-RISM (with explicit electrolyte). This is significantly more expensive and has its own convergence pitfalls.
- **U_SHE conversion** — DFT energies are referenced to vacuum/Fermi; converting to U vs. SHE requires the absolute potential of SHE (−4.44 V vs. vacuum) and a careful definition of the slab Fermi level reference (typically via a water-layer dipole shift).
- **Environ may not be in stock QE module** — verify before Phase 5; if absent, build QE with the Environ patch locally on the cluster.

### Resources to bookmark

- Quantum ESPRESSO documentation: <https://www.quantum-espresso.org/documentation/>
- PseudoDojo: <http://www.pseudo-dojo.org/>
- ASE QE calculator docs: <https://wiki.fysik.dtu.dk/ase/ase/calculators/espresso.html>
- Environ module: <https://environ.readthedocs.io/>
- Nørskov CHE paper: J. Phys. Chem. B 108, 17886 (2004) — the canonical reference for CHE.
