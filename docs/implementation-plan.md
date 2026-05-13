# Implementation Plan

Phased roadmap for the copper-oxide-dft project. Each phase has explicit deliverables and a success criterion; the project can stop after any phase with a useful, publishable result.

> See [project.md](project.md) for scope and [ground_truths.md](ground_truths.md) for methodology decisions and known gotchas.

## Methodology decisions (locked early)

| Choice | Value | Rationale |
|---|---|---|
| Functional | PBE + Hubbard U on Cu 3d | Standard for Cu oxides; pure PBE underestimates band gaps and gets energetics wrong |
| U value | Literature (~4 eV) initially; verify via `hp.x` linear response | Cheap to start, validate later |
| Pseudopotentials | PseudoDojo PBE PAW (Cu, O, H) | Modern, well-tested, free, optimized for QE |
| k-points (bulk) | Γ-centered, converged per system | 8³ for bulk Cu is a typical start |
| k-points (slab) | Scale x,y by inverse lateral, kz=1 | Standard slab convention |
| Cutoff (Ry) | `ecutwfc` 60–80, `ecutrho` 8× | Converged on bulk Cu |
| Smearing | Marzari–Vanderbilt cold, σ=0.02 Ry | Required for metallic Cu |
| Spin | `nspin=2` for CuO and O-containing slabs; off for pure Cu/Cu₂O | CuO is AFM |
| Slab geometry | 4 layers, bottom 2 fixed at bulk, 15 Å vacuum | Cost/accuracy compromise |
| Dipole correction | On for asymmetric slabs (`tefield`, `dipfield`) | Removes spurious slab–slab interaction |
| Solvation | vacuum → Environ implicit → explicit H₂O → ESM-RISM | Layered from cheap to expensive |
| Potential | CHE post-processing → ESM-FCP constant-potential | Layered |
| Compute | ORNL Andes (CPU) first; Frontier (GPU) for production | Andes is simpler to debug |

---

## Phase 0 — Toolchain setup (week 1)

**Goal:** End-to-end pipeline working with a trivial calculation.

**Steps:**
1. Verify ORNL HPC accounts (Andes, Frontier). Confirm QE module exists (`module avail quantum-espresso`).
2. Download PseudoDojo PBE pseudopotentials for Cu, O, H; commit to `pseudopotentials/` (NB: check licensing before committing — may need to gitignore and document download path).
3. Rename `src/package_name/` → `src/cuox_dft/`. Update `pyproject.toml` accordingly.
4. Add dependencies: `ase`, `pymatgen`, `numpy`, `matplotlib`, `click`.
5. Write a minimal smoke-test script: bulk Cu SCF, single k-point, 30-second runtime. Run on a login node and on a single compute node via SLURM.
6. Set up directory convention: `runs/<system>/<calc-type>/` for QE inputs and outputs.

**Deliverable:** Running `pytest` passes; running the smoke-test script returns a converged total energy for bulk Cu.

**Success criterion:** Can produce a QE input file from Python, submit it to Andes, and read the output back into Python.

---

## Phase 1 — Bulk Cu convergence (week 1–2)

**Goal:** Establish converged QE parameters for the Cu system. Teach yourself what each parameter does.

**Steps:**
1. Implement `cuox_dft/structure_builder.py: build_bulk_cu()` using ASE.
2. Implement `cuox_dft/qe_input.py: write_relax_input()` (variable-cell `vc-relax`).
3. Lattice parameter optimization: confirm `a ≈ 3.61 Å` (experiment 3.615 Å).
4. Convergence sweeps via `cuox_dft/convergence.py`:
   - `ecutwfc` ∈ {40, 60, 80, 100} Ry
   - k-grid {6³, 8³, 10³, 12³}
   - Smearing σ ∈ {0.01, 0.02, 0.03} Ry
5. Pick the smallest values that converge total energy to <1 meV/atom.
6. Save converged settings as a JSON config consumed by later phases.

**Deliverable:** Convergence plots; locked-in `ecutwfc`, `ecutrho`, k-grid, σ for Cu.

**Success criterion:** Lattice parameter matches experiment to <0.5%.

---

## Phase 2 — Bulk Cu₂O and CuO + Hubbard U (week 2–3)

**Goal:** Get oxide bulks right. Calibrate DFT+U.

**Steps:**
1. Structure: Cu₂O cuprite (Pn-3m, a=4.27 Å) — non-magnetic.
2. Structure: CuO tenorite (C2/c, monoclinic) — AFM.
3. Run with `nspin=2` for CuO; set starting magnetizations to enforce AFM ordering.
4. Sweep Hubbard U on Cu 3d: U ∈ {0, 2, 4, 6, 8} eV.
5. Compare lattice parameters and band gaps to experiment:
   - Cu₂O experimental gap: 2.17 eV
   - CuO experimental gap: 1.2–1.7 eV
6. **Optional sanity check:** compute U self-consistently via `hp.x` (Hubbard linear response).
7. Pick a single U value to use throughout the project; document in `ground_truths.md`.

**Deliverable:** Locked-in U value; bulk Cu₂O and CuO lattice + electronic structure validated.

**Success criterion:** Band gaps within ~30% of experiment, lattice parameters within ~2%.

---

## Phase 3 — Surfaces in vacuum (week 3–5)

**Goal:** Relaxed clean surface structures and surface energies.

**Steps:**
1. Cu(111) slab: 4–6 layer convergence test; pick 4 layers.
2. Lateral supercell: 3×3 or 4×4 — choose based on adsorbate coverage you want.
3. Vacuum convergence: 10, 12, 15, 18 Å. Pick smallest that converges surface energy to <0.01 eV/Å².
4. Dipole correction on asymmetric slabs.
5. Cu₂O(111) and CuO(111) terminations — multiple are possible per facet; relax each, pick lowest-energy.
6. Adsorbed O and OH on Cu(111) at 1/4, 1/2, 3/4, 1 ML coverage.
7. (Optional) Cu₂O thin film on Cu(111) — ~17% lattice mismatch, requires coincident supercell construction.

**Deliverable:** Library of relaxed surface structures + surface energies, in vacuum.

**Success criterion:** Cu(111) surface energy ≈ 1.3 J/m² (literature). Adsorbed O binding energy on Cu(111) ≈ −4.5 to −5 eV vs. ½O₂ (literature).

---

## Phase 4 — CHE Pourbaix diagram (week 5–6)

**Goal:** First-pass stability diagram across (U, pH). The "cheap" electrochemistry answer.

**Steps:**
1. Implement `cuox_dft/che.py`:
   - Free-energy formula: `ΔG(U, pH) = ΔE_DFT + ΔZPE − TΔS − neU + n·k_B T·ln(10)·pH` for reactions involving n proton-electron transfers.
   - ZPE and TΔS for adsorbed O/OH from literature tables (Nørskov values are widely used).
2. Compute Gibbs free energy of each surface (clean Cu, Cu₂O-covered, CuO-covered, O-covered, OH-covered) as function of (U, pH).
3. Build the Pourbaix-like diagram: at each (U, pH), find the lowest-G surface.
4. Compare to experimental Cu Pourbaix diagram (Pourbaix 1974 atlas).

**Deliverable:** First scientific figure — (U, pH) stability map for Cu/Cu₂O/CuO/Cu-O/Cu-OH.

**Success criterion:** Phase boundaries qualitatively match experimental Pourbaix data.

---

## Phase 5 — Implicit solvation with Environ (week 6–7)

**Goal:** Capture water's electrostatic screening.

**Steps:**
1. Confirm Environ is available with the ORNL QE build (may need a local rebuild — check first).
2. Re-run Phase 3 structures with Environ implicit water (ε=78.4, default electrolyte concentration zero).
3. Compare to vacuum: surface energies shift; phase boundaries in the Pourbaix diagram move.
4. Re-do Phase 4 with implicit-solvated free energies.

**Deliverable:** Solvated Pourbaix diagram + comparison to vacuum version.

**Success criterion:** Implicit solvation shifts surface energies in the expected direction (more polar surfaces stabilized).

---

## Phase 6 — Explicit water layer (week 7–10)

**Goal:** Include water structure and H-bonding.

**Steps:**
1. Build 1–2 explicit water layers (~24–48 H₂O) on top of each surface.
2. Pre-equilibrate water with classical MD (ASE LAMMPS interface) or short DFT-MD (~1 ps).
3. Full relaxation in QE (spin-polarized).
4. (Optional) Short AIMD trajectory (5–10 ps) to sample water orientations; analyze with `cuox_dft/analysis.py`.

**Deliverable:** Realistic interface structures with water.

**Success criterion:** Water layer remains intact (no dissociation unless physically expected) and shows reasonable H-bond network.

---

## Phase 7 — Constant-potential DFT (week 10–14)

**Goal:** Actually apply potential (not just shift energies). Required for the stated end goal.

**Steps:**
1. Confirm ESM and ESM-RISM modules in ORNL's QE build.
2. Choose ESM boundary condition (`bc1` for symmetric, `bc2` for asymmetric/electrochem).
3. Implement FCP (fictitious-charge potentiostat): iteratively adjust total charge so that the Fermi level aligns with the target U vs. SHE.
4. Reference electrode: use a water-layer dipole shift to convert calculated Fermi level to absolute (then to SHE: U_SHE ≈ −4.44 V vs. vacuum).
5. Sweep U ∈ {−1, −0.5, 0, +0.5, +1} V; re-equilibrate water and surface at each.
6. Compare to CHE results from Phase 4 — large discrepancies are interesting science.

**Deliverable:** Surface free energies at each true applied potential.

**Success criterion:** Workflow converges at each U; results show smooth U-dependence.

---

## Phase 8 — Reconstruction studies (week 14+)

**Goal:** The research payoff. Identify potential-driven reconstructions.

**Steps:**
1. Compare relaxed structures across the U sweep. Look for: cation displacements, step formation, oxide growth/dissolution, missing-row reconstructions.
2. Build candidate reconstructed structures (informed by experiment if known).
3. Constant-potential AIMD at key U values (Frontier territory — expensive).
4. Free-energy differences between candidates: nudged elastic band (NEB) for barriers if needed.

**Deliverable:** Predicted reconstruction(s) with associated potentials; structural model.

**Success criterion:** Scientifically novel and defensible result.

---

## Python package skeleton (built incrementally)

```
src/cuox_dft/
├── __init__.py
├── structure_builder.py    # ASE-based: bulk_cu, cu_slab, oxide_overlayer, adsorbates
├── qe_input.py             # pw.x input file generation with project defaults
├── submit.py               # SLURM script generation for Andes/Frontier
├── parse.py                # Read pw.x output: energies, forces, magnetization, E_F
├── convergence.py          # Sweep helpers (cutoffs, k-points, smearing)
├── che.py                  # Computational Hydrogen Electrode post-processing
├── analysis.py             # Pourbaix construction, plotting
├── config.py               # JSON-backed converged-parameter store
└── cli.py                  # Click CLI
```

Each phase adds the modules it needs; we don't write all of this upfront.

---

## Risks and known gotchas

- **DFT+U value sensitivity** — different U values give qualitatively different oxide stabilities. Mitigation: report results for a range of U values; document choice.
- **CuO AFM ordering** — starting magnetizations matter; wrong AFM order can be a local minimum. Mitigation: try multiple starting configurations.
- **Slab convergence** — fewer than 4 layers gives wrong surface energies for Cu. Mitigation: convergence test in Phase 3.
- **Environ availability at ORNL** — may not be in the stock QE module. Mitigation: build locally with Environ patch if needed (Phase 5 prerequisite).
- **Constant-potential convergence** — FCP iterations can fail to converge near band edges or at strong fields. Mitigation: start with small charges, mix charge slowly.
- **HPC queue waits** — Frontier queue can be days. Mitigation: develop on Andes; reserve Frontier for production runs only.

## Decision log

Record significant methodology decisions and parameter choices here (with date and rationale) as the project progresses.

- _YYYY-MM-DD — example entry_
