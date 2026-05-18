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
- HPC: **ORNL Frontier** (AMD MI250X GPUs) is the production target. Andes (CPU) is available as a debugging fallback.

### 2026-05-13: Cu oxide DFT gotchas (non-obvious)

- **CuO is antiferromagnetic** — `nspin=2` with explicit starting magnetizations is mandatory; non-magnetic CuO is qualitatively wrong (wrong band gap, wrong lattice).
- **Cu 3d electrons need DFT+U** — PBE underestimates Cu₂O band gap (~0.5 eV vs. 2.17 eV experiment). Typical U on Cu d is 4–7 eV (Mosey/Carter ~4 eV is a common literature pick).
- **Metallic Cu requires smearing** — without it, SCF will not converge or will give wrong forces. Marzari–Vanderbilt is the safe choice; do not use Gaussian for metals.
- **Cu₂O on Cu(111) has ~17% lattice mismatch** — needs a coincident supercell, not a simple ×n superlattice.
- **CHE is post-processing, not DFT** — the Nørskov computational hydrogen electrode shifts free energies *after* the DFT calculation by −eU − k_B T·ln(10)·pH; the underlying DFT is neutral. Good for stability diagrams, does not produce the *structure* at a given potential.
- **Constant-potential ≠ CHE** — to actually study reconstruction at potential, you need ESM-FCP (charged slab + counter-electrode) or ESM-RISM (with explicit electrolyte). This is significantly more expensive and has its own convergence pitfalls.
- **U_SHE conversion** — DFT energies are referenced to vacuum/Fermi; converting to U vs. SHE requires the absolute potential of SHE (−4.44 V vs. vacuum) and a careful definition of the slab Fermi level reference (typically via a water-layer dipole shift).
- **Environ may not be in stock QE module** — verify before Phase 5; if absent, build QE with the Environ patch locally on the cluster.

### 2026-05-13: Frontier SLURM conventions

- **Node layout:** 1 AMD EPYC 7A53 (64 cores) + 4 MI250X (8 GCDs total). Standard MPI layout is 8 ranks/node (1 per GCD), 7 cores/rank. The 64th core is reserved for the OS.
- **Cray module order:** `module purge` → `PrgEnv-{gnu,cray,amd}` → `rocm` → application (`quantum-espresso`). Order matters — application modules load AGAINST whatever toolchain is current.
- **GPU-aware MPI:** export `MPICH_GPU_SUPPORT_ENABLED=1` before `srun`. Without it, Q-E falls back to CPU↔GPU copies and burns most of the GPU speedup.
- **srun bindings:** `--gpus-per-task=1 --gpu-bind=closest` is the standard incantation for one rank per GCD.
- **SBATCH directives:** `--gpus-per-node=8` (one per GCD) + `-c 7` (cores per task). Frontier `batch` partition has a 2 h walltime cap for small node counts.
- **Q-E GPU build status:** the AMD GPU port is younger than the CPU code; smoke-test new system types against a known reference before trusting energies.

### 2026-05-14: Phase 4 CHE Pourbaix — implementation choices and validation

Implemented the Computational Hydrogen Electrode (CHE) Pourbaix construction
for solid Cu / Cu₂O / CuO phases ([che.py](../src/copper_oxide_dft/che.py),
[pourbaix.py](../src/copper_oxide_dft/pourbaix.py)). Decisions worth not
re-deriving:

- **Reservoir convention**: per-Cu free energy referenced to bulk Cu(metal)
  and H₂O(l). Oxygen chemical potential follows from H₂O ⇌ O + 2(H⁺ + e⁻):
  `μ(O) = μ(H₂O) − μ(H₂) + 2·eU + 2·k_BT·ln10·pH`. This is the standard
  Hansen/Nørskov/Persson formulation; do NOT switch to an O₂-based reservoir
  without re-deriving signs (and O₂ has the notorious DFT triplet error of
  ~0.4 eV anyway).
- **Per-Cu indexing**: each phase reports ΔG normalized by Cu atom count.
  At any (U, pH), the stable phase is the one with the minimum ΔG_per_Cu.
  Cu metal gives ΔG_per_Cu = 0 by construction.
- **Slopes**: Cu₂O has slope -1 eV/V vs. U (n_O/n_Cu = 1/2 × -2); CuO has
  slope -2 eV/V. Steeper slope = needs higher U to stabilize. This is why
  CuO appears only at high U + high pH in the diagram.
- **Default U value (Hubbard U on Cu 3d)**: 4.0 eV
  ([DEFAULT_HUBBARD_U_CU_3D_EV](../src/copper_oxide_dft/qe_input.py)). Mosey
  & Carter pick; in the typical literature range for Cu oxides. Plan: refine
  via `hp.x` linear response in Phase 2 before claiming any quantitative
  number.
- **ZPE / TΔS literature defaults** (at 298.15 K, used when the user passes
  only DFT total energies): H₂ ZPE = 0.27 eV, TΔS = 0.40 eV; H₂O ZPE = 0.56
  eV, TΔS = 0.67 eV (H₂O entropy is the gas-at-0.035-bar convention so
  μ(H₂O) approximates liquid). Source: Nørskov 2004 (PCCP 10, 3722 supp
  tables).
- **AFM CuO species splitting**: ASE's QE writer splits Cu into two species
  ("Cu" + magmom=+1, "Cu1" + magmom=-1) when per-atom magmoms are
  heterogeneous. Both sub-species need the SAME Hubbard U; the
  [`spin_and_hubbard_overrides`](../src/copper_oxide_dft/qe_input.py) helper
  mirrors ASE's algorithm so `Hubbard_U(1)` AND `Hubbard_U(2)` both get
  emitted. We discovered this empirically the first time tests ran; if you
  hand-roll a QE input bypassing the helper, remember to duplicate the U
  term for both Cu sublattices.

**Validation (literature ΔG_f defaults, NIST 298 K)**: with experimental
formation free energies plugged into the CHE machinery, the resulting
diagram reproduces the textbook Cu Pourbaix qualitatively — Cu metal at
reducing potentials, Cu₂O in a narrow band, CuO at high U / high pH, slopes
~-59 mV/pH-unit. The literature-default answer at (U = -0.4 V SHE, pH 7) is
**Cu(metal) stable**, with Cu₂O ΔG_per_Cu = +0.45 eV and CuO ΔG_per_Cu =
+1.09 eV. This is consistent with the experimental observation that native
copper oxide is electrochemically reduced under cathodic polarization in
neutral solutions.

**Limitations / known absences (Phase 4 is intentionally limited)**:
- Solid-only — no Cu²⁺(aq), no HCuO₂⁻(aq), no Cu(OH)₂. The experimental
  Pourbaix has an active corrosion region at low pH that ours treats as a
  Cu(metal) region. Adding it requires aqueous-ion energetics and an
  activity assumption.
- Bulk only — no surface termination effects, no adsorbed OH or O, no slab
  energetics. These enter in Phase 4 v2 (adsorbates on Cu(111)).
- DFT energies are placeholders (literature ΔG_f) until Phase 1-2 Frontier
  runs land. Use `make-pourbaix-inputs` to generate the QE jobs, then pass
  `--energies <json>` to the `pourbaix` command.

### 2026-05-14: Phase 1+2 Python tooling — closing the loop

Added the analyze/aggregate/config layer so the Pourbaix CLI can consume real DFT+U energies without hand-editing JSON. End-to-end flow once Frontier jobs finish:

```text
make-pourbaix-inputs ROOT         # write 5 pw.in (Cu, Cu2O, CuO, H2, H2O)
make-slurm ROOT --account=...     # wrap each in submit.sh
# (submit + wait on Frontier)
aggregate-pourbaix-energies ROOT --out energies.json   # parse pw.out tree
pourbaix --u -0.4 --ph 7 --energies energies.json      # produces real ΔG
```

**Per-formula-unit normalization** (`aggregate-pourbaix-energies`): pw.x reports total energy per cell. We divide by formula-units-per-cell using the conventional cells emitted by `build_bulk_*`: bulk_cu (1 atom = 1 f.u.), bulk_cu2o (6 atoms = 2 f.u.), bulk_cuo (8 atoms = 4 f.u.). Molecules are one f.u. each. Skipping this division silently scales Cu2O/CuO ΔG by 2× / 4× — easy to miss because the Pourbaix diagram still has the right topology with wrong slopes.

**Convergence-test semantics** (`analyze_sweep` in [analysis.py](../src/copper_oxide_dft/analysis.py)): "smallest converged value" excludes the largest sweep point itself. A single value can't prove its own convergence; if only the asymptote qualifies, the analyzer returns `None` and the `sweep-analyze` CLI exits non-zero. The user must extend the sweep upward. This is the difference between "I have a number" and "I have a defensible number."

**Per-atom energy threshold** (`DEFAULT_CONVERGENCE_THRESHOLD_MEV_PER_ATOM = 1.0`): matches the Phase 1 success criterion. Total-energy comparison would fail across system sizes (a tighter cutoff costs more meV for a bigger cell); per-atom keeps the threshold meaningful when the same analyzer is reused for slabs in Phase 3.

**Hubbard-U sweep** is now a first-class option in [convergence.py](../src/copper_oxide_dft/convergence.py): `sweep_convergence(param="hubbard_u", values=[0,2,4,6,8])` writes one pw.in per U value, routed through `spin_and_hubbard_overrides` so AFM CuO's two Cu sub-species both receive the U term. Directory labels use `0p00` / `4p00` / `6p00` to keep filesystem paths clean.

**hp.x input writer** ([qe_input.write_hp_input](../src/copper_oxide_dft/qe_input.py)): emits a minimal `&INPUTHP/` namelist for self-consistent Hubbard-U linear response. Critical detail: `prefix` here MUST match the parent SCF's `CONTROL.prefix` or hp.x can't find the saved wavefunctions. We have not exercised this against a real Frontier QE build yet — verify the namelist key names against the cluster's hp.x version before relying on the result.

**ProjectConfig** ([config.py](../src/copper_oxide_dft/config.py)): JSON-backed store for "what's locked in for system X". Schema is forward-compatible — unknown keys round-trip cleanly, so Phase 3 slab parameters (vacuum width, layer count) can land later without a schema bump. `schema_version` is checked on load; bump it only on incompatible changes.

### 2026-05-14 (later): Phase 3–8 Python scaffolding (overnight run)

Landed Python-only scaffolds for every remaining phase so the user can return next week with all of the prep work done and only the Frontier-side execution left.

**Phase 3 (surfaces in vacuum)** — [structure_builder.py](../src/copper_oxide_dft/structure_builder.py):
- `build_cu111_slab(layers, supercell, vacuum_ang, fix_bottom_layers)` — wraps ASE's `fcc111`, applies `FixAtoms` to the bottom layers. **ASE vacuum quirk**: the `vacuum` argument adds the requested thickness *on each side*, so the z-cell grows by 2×vacuum when you double it. Tests assume this.
- `build_cu2o_111_slab(layers, supercell, …)` and `build_cuo_111_slab(layers, supercell, …)` — minimal ASE-`surface`-based builders. They return *a* (111) termination, not the lowest-energy one; verify via `inspect` before submitting. Termination optimization is a Phase 3 finding, not a baked-in default.
- `add_oxygen_adsorbates(slab, coverage_ml, site, adsorbate)` — picks `round(coverage_ml * n_surface)` top-layer atoms and places O or OH at top/bridge/fcc/hcp sites. **Round-to-zero fails loudly**: requesting 1/9 ML on a 2×2 cell raises instead of silently dropping the adsorbate.
- `surface_energy_ev_per_a2(slab_E, bulk_E_per_atom, n_atoms, area, n_surfaces)` — `n_surfaces=2` for symmetric slabs, `1` for dipole-corrected asymmetric. Cu(111) literature is ~0.08 eV/Å² (≈ 1.3 J/m²).

**Phase 4 v2 (adsorbate Pourbaix)** — [che.py](../src/copper_oxide_dft/che.py) + [pourbaix.py](../src/copper_oxide_dft/pourbaix.py):
- `AdsorbateState` dataclass: a coverage state (n_O, n_OH) on a fixed Cu(111) supercell. ZPE/TS for adsorbates go in this object's `zpe_ev` / `ts_ev` fields, not in the reservoir.
- `adsorbate_state_relative_free_energy_ev(state, clean, refs, U, pH)` returns ΔG of the covered surface relative to the clean reference using `ΔG = G_state − G_clean − n_O·μ(O) − n_OH·μ(OH)`. **Critical sign convention** — there is *no* extra CHE shift on top because the U/pH dependence is already inside μ(O) and μ(OH). Adding one would double-count.
- `adsorbate_phase_diagram(states, clean, refs, …)` reuses the bulk `PourbaixDiagram` shape so plotting works unchanged. **Limit**: all states must share the same supercell (absolute energies don't subtract cleanly across cells); per-area normalization would change that and is deferred.

**Phase 5 (Environ implicit solvation)** — [environ.py](../src/copper_oxide_dft/environ.py):
- `write_environ_input(out_path, environ_type='water', …)` emits a complete `environ.in` with the &ENVIRON, &BOUNDARY, &ELECTROSTATIC namelists. Defaults: water (ε=78.36), electronic cavity (Andreussi), parabolic PBC correction along z (pbc_dim=2).
- **Requires the Environ-patched QE build**, not stock. Verify it's available on Frontier before relying on this — if absent we need a local QE rebuild with the Environ patch.

**Phase 7 (ESM-FCP constant-potential DFT)** — [qe_input.py](../src/copper_oxide_dft/qe_input.py):
- `fcp_overrides_for_potential(u_she_v, esm_bc='bc2', …)` builds the override dict that combines &CONTROL.lfcp=.true., &SYSTEM.assume_isolated='esm' + esm_bc, and &FCP.fcp_mu.
- **U → fcp_mu conversion**: `fcp_mu (Ry) = -(SHE_absolute + U) / EV_PER_RYDBERG`. `SHE_ABSOLUTE_POTENTIAL_V = 4.44` (Trasatti). The function exposes this as a parameter so a different convention (Hansen 4.28, Kelvin 4.60, etc.) can be plugged in.
- **Sign sanity**: more positive U pulls electrons out → fcp_mu becomes more negative (deeper below vacuum). Test enforces this.
- Composes cleanly with `spin_and_hubbard_overrides` by merging namelist dicts (caller-side merge — the helpers do not auto-combine, to keep responsibilities single).

**Phase 6 prep (explicit water layer)** — [structure_builder.py](../src/copper_oxide_dft/structure_builder.py):
- `add_explicit_water_layer(slab, n_waters, height_ang, layer_thickness_ang, seed)` distributes `n_waters` H₂O molecules on a near-square grid above the slab top, with random orientations driven by a deterministic seed. **Starting guess only** — production runs need MD pre-equilibration (classical or short AIMD). The seed makes runs reproducible without baking in a specific water arrangement.

**Phase 8 prep (NEB)** — [neb.py](../src/copper_oxide_dft/neb.py):
- `write_neb_input(out_path, endpoints, n_intermediate_images, …)` emits a `neb.x` input with two pinned endpoints and intermediate images filled in by QE's own interpolation. Defaults match the QE NEB tutorial: Broyden optimizer, climbing-image auto-switch, k_min=0.1/k_max=0.3 Ry/Bohr spring bounds.
- Endpoint mismatch (atom count or formula) raises rather than producing a malformed input that QE would only complain about hours into a run.

**Cross-cutting test coverage**: 165 tests, 97% line coverage across 12 modules. Ruff clean. The full Phase-4 user story (bulk Pourbaix end-to-end) and the Phase-4 v2 surface story both run on synthetic inputs in CI without Frontier.

**Known sharp edges for the next session**:
1. The Environ binary on Frontier is unverified — `write_environ_input` will produce a syntactically valid `environ.in` but the *patched* QE that consumes it may not be installed. First step in Phase 5 production: `module avail` to check, build locally if missing.
2. The `hp.x` namelist key names have not been verified against ORNL's specific QE version; sanity-check on a small Cu2O bulk before scaling up.
3. ESM-FCP convergence at large |U| can fail spectacularly. Start with U near 0 and walk outward.
4. The oxide(111) slab builders return *a* termination, not the lowest-energy one. Cleave-position optimization is a manual exercise; the implementation plan calls it out.

### 2026-05-18: Real experimental system — THF / EtOH / Ag/AgCl / U = −0.8 V

The lab system this project is modelling computationally:

- **Solvent**: tetrahydrofuran (THF), ε_static = 7.52 at 298 K (CRC handbook).
- **Proton donor**: 1 % ethanol (EtOH) in THF. The proton reservoir for any PCET step is EtOH ⇌ EtO⁻ + H⁺, *not* H₂O.
- **Reference electrode**: Ag/AgCl. Treated computationally with **absolute potential 4.64 V vs vacuum** (= 4.44 V SHE + 0.197 V Ag/AgCl in sat. KCl).
- **Target potential**: U = −0.8 V vs Ag/AgCl (≈ −0.997 V vs SHE; cathodic regime).
- **Surface**: Cu(111) with O adsorbates as a proxy for CuO/Cu(111) until a coincident-supercell builder lands (Phase 8 task).

**Implications for the workflow** (see [startup-cuo-cu-nonaqueous.md](startup-cuo-cu-nonaqueous.md) for the full walkthrough):

1. **Phase 4 aqueous Pourbaix is skipped.** The CHE machinery in [che.py](../src/copper_oxide_dft/che.py) and [pourbaix.py](../src/copper_oxide_dft/pourbaix.py) hard-codes the H₂O reservoir and uses pH as an axis. Running it on non-aqueous data yields a syntactically valid, scientifically meaningless diagram. Adapting `che.py` to an EtOH proton reservoir is a future task; the immediate path is Phase 3 surface energetics → Phase 7 ESM-FCP.
2. **ESM-FCP calls take `she_absolute_v=4.64`.** The kwarg name in [`fcp_overrides_for_potential`](../src/copper_oxide_dft/qe_input.py) is historical — it really means "absolute potential of whatever reference you're using". Worked: U = −0.8 V → μ_F = −3.84 eV vs vacuum → fcp_mu = −0.282 Ry. Lock this in every call site and never mix references mid-project.
3. **Environ defaults must be overridden.** Pass `static_permittivity=7.52` AND `environ_type='input'` (NOT `'water'`) to `write_environ_input`. Staying on `environ_type='water'` while overriding the permittivity silently keeps Environ's built-in water parameters. THF (ε=7.52) screens ~10× less than water (ε=78.36), so implicit-solvation shifts on this system are modest — order-of-coverage stability at U = −0.8 V is unlikely to flip between vacuum and implicit-THF.

**Caveats worth carrying forward**:

- **Ag/AgCl in non-aqueous is a pseudo-reference.** The 4.64 V absolute is the aqueous-sat-KCl value; in THF the liquid-junction potential at the cell shifts it by O(0.1–0.3 V) with direction depending on cell construction. For quantitative U values, plan to calibrate against internal Fc/Fc⁺ (Connelly & Geiger, *Chem. Rev.* 1996 is the canonical conversion source) and re-derive `she_absolute_v` from the measured offset. For preliminary work, 4.64 V is defensible — just don't quote it as exact.
- **EtOH proton donor only matters for explicit chemistry.** Pure ESM-FCP at fixed U doesn't care about the proton source — we set the Fermi level directly. The EtOH reservoir becomes load-bearing if/when we extend `che.py` for PCET mechanism analysis or build explicit-EtOH overlayers (Phase 6 territory).

### 2026-05-18: Hardware path — DGX Spark prototype → Frontier production

The compute target for this project is a two-stage pipeline rather than a single cluster:

- **Prototype**: NVIDIA **DGX Spark (GB10 Grace+Blackwell)**, a single-node workstation. ARM CPU + Blackwell-class GPU + 128 GB unified memory. No SLURM. Run `pw.x` directly via `mpirun -n 1` plus OpenMP on the Grace side.
- **Production**: **ORNL Frontier** (AMD MI250X, 8 GCDs/node). SLURM. Conventions already documented in the Frontier section above.

**Key build/runtime differences from the Ubuntu-CPU and Frontier paths the rest of this doc assumes**:

- DGX Spark needs a **CUDA-aware** QE build (NVHPC SDK + CUDA + cuBLAS/cuFFT), not the AMD HIP build. Configure flag: `--with-cuda-cc=120` (Blackwell — verify against `nvidia-smi --query-gpu=compute_cap`). The apt QE package is CPU-only on ARM.
- `make-slurm` is **not used** for DGX Spark runs — no scheduler. A trivial `qe-run <dir>` shell wrapper that does `cd <dir> && mpirun -n 1 pw.x -in pw.in > pw.out` is the equivalent. Could become a `make-runner` CLI sibling to `make-slurm` later.
- GB10 wall times **do not extrapolate to Frontier**. Blackwell single-GPU and MI250X (×8 GCDs/node) have different peak FLOPS, memory bandwidth, and MPI scaling. Benchmark a small case on Frontier before sizing production jobs.

### Resources to bookmark

- Quantum ESPRESSO documentation: <https://www.quantum-espresso.org/documentation/>
- PseudoDojo: <http://www.pseudo-dojo.org/>
- ASE QE calculator docs: <https://wiki.fysik.dtu.dk/ase/ase/calculators/espresso.html>
- Environ module: <https://environ.readthedocs.io/>
- Nørskov CHE paper: J. Phys. Chem. B 108, 17886 (2004) — the canonical reference for CHE.
