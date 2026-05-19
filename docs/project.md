# Project Overview

## Project Name

**copper-oxide-dft** — Python pipeline for DFT calculations of copper oxide phases on copper surfaces under electrochemical conditions.

## Purpose

Predict the structure and stability of copper oxide phases (Cu, Cu₂O, CuO, suboxides) on Cu(111) under cathodic polarization in **non-aqueous** electrolyte (THF + 1 % EtOH) at U = −0.8 V vs Ag/AgCl, using Quantum ESPRESSO. The headline scientific question is *why does a copper-oxide overlayer **remain** on Cu(111) under conditions where the aqueous Pourbaix predicts it should reduce?* The answer requires structural discovery — finding the minimum-Ω surface across the full Cu–O composition space — rather than ranking a hand-built coverage ladder. See [ml-gcgo-pivot.md](ml-gcgo-pivot.md) for the active workflow.

The earlier aqueous-Pourbaix scope (Cu / Cu₂O / CuO stability across (U, pH), built on the Computational Hydrogen Electrode) is implemented in [che.py](../src/copper_oxide_dft/che.py) and [pourbaix.py](../src/copper_oxide_dft/pourbaix.py); it is intact but **not the central question** for this iteration. Use it as a sanity-check reference, not the production answer.

## Target Users

Primarily the project owner (computational scientist at ORNL, new to DFT). Designed so the workflow is reproducible by other group members and the analysis steps are accessible to experimentalists.

## Core Functionality

1. **Structure generation** — bulk, slab, and oxide/Cu interface structures via ASE (Cu(111), Cu₂O(111), CuO(111), adsorbed O/OH at varying coverage). Used as seed cells for the MLIP-GCGO box-sampling stage.
2. **QE input generation** — `pw.x` input files with project-wide defaults (PBE + Hubbard U on Cu 3d, Marzari–Vanderbilt smearing, PseudoDojo PBE PAW pseudopotentials, dipole correction).
3. **MLIP-GCGO pipeline (central engine)** — `src/copper_oxide_dft/ml/`: box-sampling perturbed seed cells, MACE-MP-0 fine-tuning on QE-relaxed structures, grand-canonical genetic algorithm over μ_O on a Cu(111) substrate, ensemble post-processing to per-x_O minima.
4. **HPC job submission** — two-stage path. DGX Spark (NVIDIA Grace + Blackwell) handles dataset generation, MACE training, and GCGA inference; ORNL Frontier (AMD MI250X) handles the top-K ESM-FCP rerank at U = −0.8 V. SLURM scripts via `make-slurm`.
5. **Output parsing** — energies, forces, magnetizations, Fermi levels, FCP electron counts from QE output.
6. **Electrochemistry post-processing** — Computational Hydrogen Electrode + aqueous Pourbaix (intact, out-of-scope for the non-aqueous question). Grand-canonical Ω(U, structure) reranking for the MLIP-GCGO output (in scope).
7. **Experimental observable bridge** — `ml/sld.py` converts each ensemble member to a neutron-reflectometry SLD profile with bulk-Cu normalization. This is what makes the prediction falsifiable against the experimental NR data.
8. **Refinements available, not in the critical path** — Environ implicit solvation (single-point Environ-THF on the predicted winner); NEB (kinetic barrier from existing CuO to the predicted thermodynamic ground state); explicit EtOH overlayer (only if PCET mechanism analysis becomes a question).

## Input/Output

**Input:**
- High-level configuration (system type, surface facet, μ_O range, target U).
- Pseudopotential files (PseudoDojo PBE PAW for Cu, O, H).
- HPC target (DGX Spark workstation for dataset/training/GCGA; Frontier for ESM-FCP reranks).
- MACE-MP-0 medium foundation-model weights.

**Output:**
- QE input/output files in a structured directory tree per calculation.
- Parsed results (JSON, HDF5): energies, forces, magnetizations, geometries, FCP electron counts.
- Fine-tuned MACE model + held-out test metrics (energy MAE, force MAE).
- GCGA ensemble: ~10⁴ phases parametrised by μ_O and x_O.
- Frontier ESM-FCP rerank: Ω(U = −0.8 V) for the top-K candidates and the single predicted winner.
- Plots: convergence curves, MACE learning curves, Ω vs x_O at U = −0.8 V, SLD vs depth overlaid on the experimental neutron-reflectometry profile.

## Example Usage

The MLIP-GCGO pipeline (in development under `src/copper_oxide_dft/ml/`):

```python
from copper_oxide_dft.config import load_config
from copper_oxide_dft.ml import box_sampling, qe_driver, curate, gcga, ensemble, fcp_rerank

# 1. Seed structures perturbed via GOCIA box-sampling
cfg = load_config("configs/converged.json")
seeds = box_sampling.generate(seed_compositions=["Cu", "Cu8O", "Cu2O", "Cu4O3", "CuO"],
                              n_per_seed=500, rattle_ang=0.2, lattice_scale=0.05)

# 2. QE relaxation on DGX Spark (vacuum, Γ-only, PBE + Hubbard U)
relaxed = qe_driver.relax_batch(seeds, config=cfg, out_root="runs/ml_dataset")

# 3. Curate (force filter, SOAP+UMAP subsample, 10:1 split, extxyz)
curate.write_extxyz(relaxed, train="cuox_train.extxyz", test="cuox_test.extxyz")

# 4. MACE fine-tuning is a shell step (scripts/finetune_mace.sh)

# 5. GCGA on Cu(111) 12-layer slab with the fine-tuned MACE potential
ensemble_path = gcga.run(model="cuox_pbe_finetuned.model",
                          substrate=gcga.cu111_substrate(layers=12, lateral=(4, 4)),
                          mu_o_range=(-7.0, -6.0),
                          biased_x_o_range=(0.32, 1.0))

# 6. Top-K ESM-FCP rerank at U = -0.8 V vs Ag/AgCl (Frontier)
fcp_rerank.emit_inputs(ensemble_path, top_k=20, u_target_v=-0.8,
                       reference_absolute_v=4.64,
                       out_root="runs/fcp_rerank_minus0p8V")
```

The earlier (aqueous-Pourbaix) CLI commands are still available for the bulk + adsorbate analyses they were built for:

```bash
copper-oxide-dft sweep --param ecutwfc --values 40,60,80,100,120 --out runs/conv_ecutwfc
copper-oxide-dft pourbaix --u -0.4 --ph 7 --energies runs/oxides/energies.json
copper-oxide-dft make-slurm runs/fcp_rerank_minus0p8V --account <project>
```

## Dependencies

- **Core**: `ase`, `pymatgen`, `numpy`, `matplotlib`
- **CLI**: `click`
- **Testing**: `pytest`, `pytest-cov`
- **MLIP-GCGO pipeline (new)**: `gocia` (structural search + GCGA driver), `mace-torch` (foundation-model fine-tuning), `dscribe` (SOAP descriptors), `scikit-learn` (Incremental PCA), `umap-learn` (2-D projection), `h5py` (ensemble store).
- **External (not pip)**: Quantum ESPRESSO with CUDA build (`pw.x`, `hp.x`, optional Environ patch, optional ESM‑RISM); PseudoDojo PBE PAW pseudopotential library; SLURM on ORNL Frontier.

## Technical Notes

- **Functional**: PBE + Hubbard U = 4.0 eV on Cu 3d. Kept across the MLIP-GCGO pivot for continuity with the converged Phase 1 settings; the calibration gap with the reference manuscript's PBEsol is documented in [ml-gcgo-pivot.md §3.1](ml-gcgo-pivot.md).
- **Spin polarization**: required for CuO (antiferromagnetic) and any O-containing surface; not needed for pure Cu or Cu₂O.
- **k-point sampling**: Γ-centred Monkhorst–Pack for bulk; Γ-only for the 100+ atom box-sampling perturbed cells; (6, 6, 1) for Cu(111) slabs in the proxy ladder.
- **Smearing**: Marzari–Vanderbilt cold smearing, σ = 0.01 Ry (locked in `configs/converged.json:bulk_cu.degauss_ry`).
- **Slab convention (proxy ladder)**: 4 layers, bottom 2 fixed, 20 Å vacuum, `nosym=True` (the FixAtoms constraint breaks the slab's inversion symmetry).
- **Slab convention (GCGA substrate)**: Cu(111) 12 layers, top 6 layers active, lateral (4×4) baseline.
- **Solvation strategy**: vacuum throughout DFT dataset / MACE training / GCGA / top-K ESM-FCP rerank. Environ-THF (ε = 7.52) as a one-off single-point on the predicted winner.
- **Potential strategy**: μ_O-parametrised GCGA in vacuum, then ESM-FCP at U = −0.8 V vs Ag/AgCl (`she_absolute_v = 4.64`) on Frontier. Grand-canonical Ω = E_DFT − μ_e · N_e ranks the top-K.
- **Compute target**: two-stage. **DGX Spark (NVIDIA Grace + Blackwell, 128 GB unified memory)** for dataset generation, MACE training, GCGA inference. **ORNL Frontier (AMD MI250X, 8 GCDs/node)** for the top-K ESM-FCP production reranks.

See [ml-gcgo-pivot.md](ml-gcgo-pivot.md) for the active workflow, [implementation-plan.md](implementation-plan.md) for the original phased roadmap, [startup-cuo-cu-nonaqueous.md](startup-cuo-cu-nonaqueous.md) for the non-aqueous setup details that survive the pivot (Phases 1, 2, 7), and [ground_truths.md](ground_truths.md) for methodology decisions and known gotchas.
