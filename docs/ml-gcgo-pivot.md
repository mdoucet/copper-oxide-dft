# MLIP-GCGO Pivot

**Status**: scope-locked 2026-05-18. Supersedes Phase 3 of
[startup-cuo-cu-nonaqueous.md](startup-cuo-cu-nonaqueous.md) as the
central structural-discovery engine. Phases 1, 2, and 7 of the existing
plan remain unchanged and are reused.

---

## 1. Scientific question (refined)

> *Why does a copper-oxide overlayer **remain** on a Cu(111) substrate
> at U = −0.8 V (vs Ag/AgCl) in THF + 1 % EtOH?*

Three sub-questions follow from the verb "remain":

1. **Thermodynamic.** At μ_e = −3.84 eV vs vacuum (corresponding to U =
   −0.8 V vs Ag/AgCl, [`fcp_overrides_for_potential`](../src/copper_oxide_dft/qe_input.py)),
   what is the minimum-Ω surface state across the full Cu–O composition
   space? If a CuO-like coordination wins, "remains" is thermodynamic.
2. **Kinetic.** If clean Cu wins thermodynamically, what is the lowest
   barrier path from an existing CuO terminus to the predicted ground
   state? The Phase 9 NEB scaffolding ([neb.py](../src/copper_oxide_dft/neb.py))
   answers this *after* Block F identifies the predicted ground state.
3. **Observable.** Does the predicted ground-state ensemble produce a
   neutron-reflectometry SLD profile that matches the experimental one?

The MLIP-GCGO pivot is *required* for sub-question 1; sub-question 2 is
follow-up work; sub-question 3 is what makes the prediction falsifiable.

## 2. Why the proxy ladder cannot answer this

The original Phase 3 approach in
[startup-cuo-cu-nonaqueous.md §4](startup-cuo-cu-nonaqueous.md) ranks four
hand-built coverages (¼, ½, ¾, 1 ML O on Cu(111)). Its own §0.2
acknowledges that 1 ML O on Cu(111) is geometrically *not* CuO (3-fold
adsorbate coordination vs 5-fold in bulk CuO). The 4-coverage scan:

- skips intermediate suboxides (Cu₈O, Cu₄O₃) where the answer may live;
- enforces a specific O site (fcc) that may not be the lowest-energy
  termination at the relevant μ_O;
- cannot find a reconstruction it wasn't told to look for.

A structural-search method (genetic algorithm over μ_O) is the
methodologically honest answer once the question is "what wins" rather
than "rank these four candidates."

## 3. Methodology (locked)

Decisions that *will* be re-questioned later need to be re-decided here
first, not silently in a script.

### 3.1 Functional and pseudopotentials

| Choice | Value | Why |
|---|---|---|
| Exchange-correlation | **PBE** | Continuity with Phase 1 (`configs/converged.json`). Switching to PBEsol mid-project would invalidate the converged `ecutwfc=100 Ry`, `a=3.6577 Å`, and AFM CuO `U=4.0 eV` decisions in [ground_truths.md](ground_truths.md). |
| Hubbard U on Cu 3d | **4.0 eV** | Mosey/Carter literature pick already used. Box-sampling will perturb stoichiometries far from CuO, but the U term applies uniformly to all Cu sites. |
| Pseudopotentials | **PseudoDojo PBE PAW** | Same as Phases 1–7. |
| `ecutwfc` | **100 Ry** | From `configs/converged.json:bulk_cu.ecutwfc_ry`. |
| Lattice `a` for seed cells | **3.6577 Å** | From `configs/converged.json:bulk_cu.lattice_a_ang` (the PBE-relaxed value). |
| k-points (perturbed cells) | **Γ-only** | Cells are 100+ atoms; matches the manuscript convention. |

**Calibration gap with the reference manuscript.** The Sandia paper used
VASP + PBEsol; we are using QE + PBE. PBEsol typically gives ~1 %
shorter metallic lattice constants and slightly different surface
energetics. The MACE model trained on our data will be *internally
consistent* but cannot be directly cross-validated against the paper's
quoted 9.8 meV/atom test MAE — only against a held-out PBE test set.
Expect our test MAE to land in the 10–20 meV/atom range; flag anything
above 30 as a training-pipeline problem.

### 3.2 Solvation strategy

Vacuum throughout the structural search; Environ only as a final
correction.

| Stage | Solvation | Reason |
|---|---|---|
| Box-sampling DFT dataset | **Vacuum** | Matches Sandia manuscript; Environ on perturbed 100+ atom cells is fragile and would add a parameter (the Environ cavity) to a problem that doesn't need it for the ranking. |
| MACE training | **Vacuum** | Inherits from the dataset. |
| GCGA structural search | **Vacuum (μ_O only)** | The genetic algorithm is fast in vacuum; running ESM-FCP inside the GA fitness function is computationally infeasible at GCGA scale. |
| Top-K ESM-FCP rerank (Frontier) | **Vacuum** | THF (ε = 7.52) screens ~10× less than water (ε = 78.36). The Ω ordering across coverages should not flip; the rerank establishes the constant-potential ground state. |
| Final winner (single point) | **Environ-THF (ε = 7.52)** | One additional run to quote the solvent-included Ω for the predicted winner. If this flips the ranking against #2 by less than ~50 meV, re-rerank the top 3 with Environ-THF; otherwise report vacuum. |

This is more conservative than running Environ throughout, and more
honest than assuming vacuum is "good enough" for the final number.

### 3.3 μ_O ↔ μ_e bridge

The GCGA in the manuscript ranks structures by the grand potential at a
fixed *oxygen chemical potential*:

    Ω_O(structure) = E_DFT(structure) − μ_O · N_O(structure)

The experimental question is fixed at a *constant electrochemical
potential* U. The two are connected via the proton–electron
equilibrium of the proton donor. For aqueous H₂O,

    H₂O ⇌ ½O₂ + 2(H⁺ + e⁻)    ⇒    μ_O = μ_H₂O − 2·μ_H⁺e⁻

For our system (EtOH proton donor) the equivalent equilibrium is

    EtOH ⇌ ½(O bound to surface) + (something containing H)

This equation is **not** closed in our current code. Three options:

1. **Treat μ_O as a free parameter** for the GCGA sweep range
   [−7.0, −6.0] eV (manuscript range, vacuum reference). Plot
   Ω vs μ_O across all candidate structures. The ESM-FCP rerank at fixed
   U then *re-evaluates* Ω = E_DFT − μ_e · N_e directly, without needing
   the μ_O ↔ μ_e equality. **This is the working assumption.**
2. **Anchor μ_O to U via the EtOH equilibrium.** Requires DFT energies
   for EtOH(g) and EtO⁻ (or EtO-bound), plus an EtOH proton-donor
   reservoir convention in `che.py`. Future work; not blocking.
3. **Anchor μ_O to U via the water equilibrium** (current `che.py`
   convention). **Don't do this** — water is not the proton donor in the
   experimental system, and using μ_H₂O introduces a 100+ meV bias that
   the rest of the workflow assumes away.

**Operational consequence**: the GCGA produces an *ensemble of
candidates* parametrised by μ_O; the constant-potential answer comes
from re-ranking that ensemble by Ω(μ_e) on Frontier. The MACE model
never needs to know about U.

### 3.4 Reference electrode

Unchanged from [ground_truths.md](ground_truths.md) 2026-05-18: Ag/AgCl
with absolute potential **4.64 V vs vacuum**. Every ESM-FCP call passes
`she_absolute_v=4.64`. Caveat about non-aqueous pseudo-reference drift
stands; calibration against Fc/Fc⁺ is a future correction.

### 3.5 Seed compositions and supercell

| Stage | Cells | Sources |
|---|---|---|
| Box-sampling seeds | Cu, Cu₈O, Cu₂O, Cu₄O₃, CuO, c-CuO | Manuscript; bulk Cu and bulk CuO come from [`build_bulk_cu`](../src/copper_oxide_dft/structure_builder.py) / [`build_bulk_cuo`](../src/copper_oxide_dft/structure_builder.py), the others built from materials-project–style structures or via GOCIA's `random_box` initializer. |
| GCGA substrate | Cu(111) 12 layers, lateral (m×n) tbd | Top 6 layers active region. Lateral size is the trade-off between μ_O resolution and wall time; start with (4×4) and revisit. |
| Active species | Cu, O | No Cu insertion in this round — only O reorganization on a fixed Cu base. Cu mobility is a Phase 8 extension. |

## 4. What survives from the existing repo

| Component | Status | Notes |
|---|---|---|
| `configs/converged.json` (Phase 1) | **Reused** | Source of ecutwfc, kpts (for non-Γ uses), degauss, lattice. |
| `build_bulk_cu`, `build_bulk_cuo`, `build_cu2o_111_slab` | **Reused** | Seed structures for GOCIA. |
| `write_pw_input`, `spin_and_hubbard_overrides` | **Reused** | Box-sampling QE input generation. |
| `fcp_overrides_for_potential` | **Reused** | Top-K ESM-FCP rerank inputs. |
| `make-slurm`, `submit.py` | **Reused** | Frontier production for Block F. |
| `parse.py` (`parse_pw_output`) | **Reused + extended** | Existing scalars + new `extxyz_from_pw_output` helper. |
| `build_cu111_slab` + `add_oxygen_adsorbates` (Phase 3 ladder) | **Retired for the central question.** Kept in-tree as a useful proxy / sanity check; not the production answer. |
| `che.py`, `pourbaix.py` (aqueous Pourbaix) | **Out of scope** for the non-aqueous question. No change needed. |
| `environ.py` | **Used once** at the very end (§3.2 final stage). |
| `neb.py` | **Out of scope until sub-question 2 (kinetic).** Available when needed. |

## 5. New code surface

All under `src/copper_oxide_dft/ml/`:

| Module | Responsibility |
|---|---|
| `box_sampling.py` | Generate perturbed structures (rattle, scale, O insert/delete, Hookean pre-opt). |
| `qe_driver.py` | Wrap ASE-Espresso calculator with the converged config; batched execution; manifest tracking. |
| `curate.py` | Force-filter, SOAP+IPCA+UMAP subsample, extxyz writers. |
| `validate.py` | Energy/force MAE on held-out test set. |
| `gcga.py` | Wrap GOCIA's GCGA on Cu(111); μ_O sweep + Gaussian-biased pass. |
| `ensemble.py` | Merge biased + unbiased; per-x_O minima extraction; HDF5 ensemble store. |
| `fcp_rerank.py` | Pick top-K; generate ESM-FCP inputs at U = −0.8 V; emit Frontier SLURM. |
| `sld.py` | 10 Å interfacial slab → SLD with bulk-Cu normalization. |

CLI surface (later, post-MVP): `copper-oxide-dft ml-sample`, `ml-curate`,
`ml-finetune`, `ml-gcga`, `ml-rerank`, `ml-sld`.

## 6. Success criteria

Success is sub-question 1 answered:

- ✅ MACE model trained with test MAE < 30 meV/atom and < 100 meV/Å on a
  PBE-flavored hold-out.
- ✅ GCGA ensemble of ≥ 10 000 phases spanning x_O ∈ [0, 1].
- ✅ Per-x_O minimum-Ω structures extracted and ESM-FCP-converged at
  U = −0.8 V on Frontier for the top-K (K ≈ 20).
- ✅ A single, defensible answer: which (x_O, structure) minimises
  Ω(U = −0.8 V) in the predicted ensemble.
- ✅ SLD profile for the winner (and the top 3) plotted against the
  experimental neutron-reflectometry data.

Sub-questions 2 (NEB kinetics) and 3 (full SLD ensemble comparison) are
explicitly *out of scope* for this iteration; they become tractable
*because* the structural search is done, but they are not part of the
"does CuO remain at −0.8 V" headline answer.

## 7. Known risks (in priority order)

1. **GOCIA / MACE installation churn on ARM.** Both packages target
   x86_64 NVIDIA primarily; aarch64 wheels may be missing. First
   measurable Block-B milestone is "the official examples run on DGX
   Spark." If installation eats more than 3 days, fall back to building
   the dataset on Frontier and porting only the MACE inference back to
   the workstation.
2. **PBE vs PBEsol calibration.** The manuscript's hyperparameters
   (batch=4, lr=0.01, 50 epochs) were tuned against PBEsol energies. If
   our PBE test MAE refuses to drop below 30 meV/atom, the first thing
   to vary is `max_num_epochs` (try 100) and `lr` (try 0.005). Don't
   change the optimizer (AMSGrad) or batch size without re-reading the
   manuscript's ablation tables.
3. **μ_O ↔ μ_e mismatch.** The vacuum-μ_O GCGA may concentrate
   candidates in a range of x_O that doesn't overlap the experimental
   regime. If post-rerank we find all top-K candidates collapse to
   one x_O bin, the GCGA bias was too narrow; widen the Gaussian-biased
   range or add a second μ_O sweep at [−5.5, −4.5].
4. **ESM-FCP non-convergence on disordered surfaces.** The Phase 7
   smoke test passed on a 4-layer (2×2) clean Cu(111). The Block F
   top-K candidates will be 12-layer (4×4) disordered cells; FCP loops
   may diverge. Mitigation: start each Frontier job at U = 0 V and
   ramp; see [ground_truths.md](ground_truths.md) "ESM-FCP convergence."
5. **Ag/AgCl pseudo-reference drift in THF.** Unchanged from
   [ground_truths.md](ground_truths.md) 2026-05-18. The reported U
   value has an O(0.2 V) systematic uncertainty until Fc/Fc⁺
   calibration lands.

## 8. Out of scope (record so we don't drift)

- **Cu mobility / coincident CuO/Cu(111) slab.** The original Phase 8
  TODO. The MLIP-GCGO pivot replaces *Phase 3*, not Phase 8. If the
  Block F winner is a high-coverage O state on Cu(111), that hints at
  the coincident slab being the right next step; if it's a
  reconstruction with substantial Cu rearrangement, the GCGA active
  species set needs to grow to include Cu.
- **Explicit EtOH overlayer.** Required only if sub-question 2 grows to
  include PCET steps. Phase 6 territory.
- **NEB kinetics.** Sub-question 2 follow-up; not part of the structural
  answer.
- **CHE / aqueous Pourbaix.** `che.py` and `pourbaix.py` are aqueous
  and stay untouched. Adapting them to EtOH is future work.

## 9. Related docs

- [machine-learned-dft.md](machine-learned-dft.md) — reference
  manuscript walkthrough this pivot is based on.
- [startup-cuo-cu-nonaqueous.md](startup-cuo-cu-nonaqueous.md) —
  pre-pivot workflow. Phases 1, 2, and 7 are still authoritative.
- [implementation-plan.md](implementation-plan.md) — 9-phase roadmap.
  Phase 3 is now superseded for the central scientific question.
- [ground_truths.md](ground_truths.md) — methodology decisions; this
  document adds the 2026-05-18 MLIP-GCGO pivot entry.
- [project.md](project.md) — scope; updated to point here.
