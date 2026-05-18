# Startup Guide: CuO on Cu(111) in Non-Aqueous Electrolyte at −0.8 V

End-to-end workflow for a specific scientific question:

> *How does Cu(111) evolve toward a CuO termination under cathodic polarization
> (U = −0.8 V) in a non-aqueous electrolyte?*

The plan: prototype the workflow on a **DGX Spark (NVIDIA GB10)** workstation,
then move the converged inputs to **ORNL Frontier** for production.

---

## TL;DR — the path

```text
Phase 1  bulk Cu convergence ............... DGX Spark, hours
Phase 2  bulk CuO + Hubbard U .............. DGX Spark, hours-overnight
Phase 3  Cu(111) + O adsorbates 1/4→1 ML ... DGX Spark, overnight per coverage
Phase 5  Non-aqueous Environ (implicit) .... DGX Spark, validation only
Phase 7  ESM-FCP at U = -0.8 V ............. Frontier production
Phase 8  CuO/Cu(111) coincident slab ....... Frontier production
```

The repo today ships full scaffolding for Phases 1–7 (CLI + helpers; see
[implementation-plan.md](implementation-plan.md)). Three pieces of new
tooling you will write as you go through this guide are flagged with
**TODO** below — the workflow tells you what they need to do, and the
test suite tells you what shape they need to take.

---

## 0. Three deviations from the baseline implementation plan

The repo's [implementation-plan.md](implementation-plan.md) and
[docs/local-workstation.md](local-workstation.md) assume **aqueous**
electrochemistry on **Frontier (AMD)** and **Ubuntu CPU** workstations.
Your workflow differs on three axes. Read this section before any
calculation.

### 0.1 Non-aqueous electrolyte ⇒ skip the aqueous Pourbaix, change the reference

The Computational Hydrogen Electrode ([che.py](../src/copper_oxide_dft/che.py))
and the Pourbaix diagram builder ([pourbaix.py](../src/copper_oxide_dft/pourbaix.py))
reference μ(H₂O) and use pH as an axis. Neither concept applies in
**THF with 1 % EtOH as proton donor**. You skip Phase 4 entirely and
go from Phase 3 surface energetics straight to Phase 7 constant-potential
DFT.

The real-system choices for this project (**commit these to
[ground_truths.md](ground_truths.md)** before doing any calculations):

| Quantity | Value | Notes |
|---|---|---|
| Solvent | **THF** | ε_static = 7.52 (CRC, 298 K). ~10× weaker screening than water; implicit-solvation shifts will be modest. |
| Proton donor | **1 % EtOH** | μ(H⁺ + e⁻) reference is EtOH ⇌ EtO⁻ + H⁺, not H₂O. Doesn't affect ESM-FCP at fixed U (we set the Fermi level directly); becomes relevant if you later do CHE-style PCET. |
| Reference electrode | **Ag/AgCl** | Absolute ≈ **4.64 V vs vacuum** (= 4.44 V SHE + 0.197 V Ag/AgCl-sat-KCl). **Caveat:** in non-aqueous solvents Ag/AgCl is a *pseudo-reference* whose true potential depends on cell construction. The rigorous fix is to calibrate against internal Fc/Fc⁺; the practical answer is to pick one number and use it consistently. |

The constant-potential helper
[`fcp_overrides_for_potential`](../src/copper_oxide_dft/qe_input.py)
exposes `she_absolute_v` as a parameter (the name is historical — it's
really "absolute potential of *whatever reference you're using*"):

```python
fcp_overrides_for_potential(-0.8, she_absolute_v=4.64)
# fermi_level_ev_vs_vacuum = -(4.64 + (-0.8)) = -3.84 eV
# fcp_mu (Ry)              = -3.84 / 13.6057 = -0.282 Ry
```

So at U = −0.8 V vs Ag/AgCl the electron chemical potential sits at
−3.84 eV vs vacuum. The same U interpreted as vs aqueous SHE would give
−3.64 eV; the 0.2 V offset is exactly the Ag/AgCl–SHE conversion. If
you ever quote U vs SHE in a paper, **subtract 0.197 V** from your
Ag/AgCl readings.

**If you later calibrate against Fc/Fc⁺** (the rigorous non-aqueous
practice), the offset is solvent-dependent — in THF it's roughly +0.50
V from Fc/Fc⁺ to Ag/AgCl (literature varies; Connelly & Geiger, *Chem.
Rev.* 1996 is the canonical source). Re-derive your `she_absolute_v`
once the calibration is in hand and update `ground_truths.md`.

### 0.2 "CuO on Cu" ⇒ start with O adsorbates, defer the coincident slab

`build_cu111_slab` and `build_cuo_111_slab` exist; a **CuO/Cu(111)
coincident supercell builder does not**. Cu(111) and CuO(111) have a
~17 % lattice mismatch, so any honest model needs a Diophantine search
over commensurate `(m×n)` Cu / `(p×q)` CuO cells minimising strain — that
is Phase 8 territory (reconstruction studies) and a substantial effort
in its own right.

For a meaningful first answer on a few weeks of DGX-Spark time, model the
CuO/Cu interface as a **Cu(111) slab with increasing chemisorbed-O
coverage**:

| Coverage | Surface chemistry | What it represents |
|---|---|---|
| 0 ML | clean Cu(111) | Cu under cathodic protection |
| 1/4 ML O fcc | Cu(111)-O dilute | Onset of oxidation |
| 1/2 ML O | Cu(111)-O ordered | Pre-Cu₂O surface |
| 3/4 ML O | Cu(111)-O saturated | Cu₂O-like coordination |
| 1 ML O | full O monolayer | CuO-like coordination |

This isn't *the* answer to "how does the surface evolve toward CuO" —
it's a tractable family of starting structures whose relative free
energies at U = −0.8 V tell you *which way the system wants to go*. At
−0.8 V you expect bare Cu to win (cathodic regime should reduce any
existing oxide); the experiment's reconstruction signal would be a
relaxed Cu surface, possibly with cation displacement, **not** O
adsorbates. If your DFT predicts something else, that's the interesting
result.

**Later** (Phase 8, when you're on Frontier): write the coincident-cell
builder and re-do the production run with a true CuO/Cu(111) interface.

> **TODO (Phase 8)**: `build_cuo_cu111_coincident_slab(...)`. A search
> over commensurate `(m,n)` Cu × `(p,q)` CuO cells minimising in-plane
> strain, followed by stacking the strained CuO slab on top of a fixed
> Cu(111) base. The test in `tests/test_structure_builder.py` for
> `build_cu2o_111_slab` is the template; the matching algorithm is the
> standard Zur-McGill construction (pymatgen has it as
> `pymatgen.analysis.interfaces.coherent_interfaces` but pulling in
> pymatgen for one function may not be worth it).

### 0.3 DGX Spark (GB10) ⇒ no SLURM, CUDA-built QE, ARM userland

GB10 is NVIDIA Grace+Blackwell (ARM CPU + Blackwell GPU, 128 GB unified
memory), a single workstation rather than a cluster. The repo's
`make-slurm` command targets ORNL clusters; on GB10 you run `pw.x`
directly via `mpirun`.

QE specifics on GB10:

- The **AMD HIP build** documented in [ground_truths.md](ground_truths.md)
  does not apply here. You need a **CUDA-aware** QE build.
- The official QE GPU port targets NVIDIA via NVHPC + CUDA + cuBLAS/cuFFT.
  Build flags: `./configure --enable-openmp --with-cuda=$NVHPC_ROOT
  --with-cuda-cc=120 --with-cuda-runtime=12.x` (compute capability `120`
  is Blackwell — verify against `nvidia-smi --query-gpu=compute_cap`).
- ARM userland: `apt install quantum-espresso` *may* work on DGX OS for
  the CPU version but the GPU port wants to be built from source against
  the NVHPC SDK that NVIDIA ships in the DGX Spark base image.
- One GB10 superchip has one Blackwell-class GPU; the typical run is
  `mpirun -n 1 pw.x ...` plus OpenMP threads on the Grace CPU side. You
  will not parallelise across MPI ranks on a single GB10.

**Section 1 below walks through the install end-to-end.** If you don't
have CUDA-aware QE working, **everything in this guide reduces to
CPU-only** — which is workable for Phases 1–2 but slow for Phase 3.

> **TODO**: `SlurmConfig.for_dgx_spark(...)` is not needed (no SLURM),
> but a small `runs/<system>/run.sh` wrapper that loads the right
> NVHPC modules and `mpirun`s `pw.x` would help. Not required for the
> workflow — typing it inline is fine — but a `make-runner` command
> sibling to `make-slurm` would be a clean future addition.

---

## 1. Install on DGX Spark

### 1.1 Base system

DGX Spark ships with DGX OS (Ubuntu-based, ARM64). Verify:

```bash
uname -m              # aarch64
lsb_release -a        # Ubuntu 22.04 or newer
nvidia-smi            # Blackwell GPU visible, driver loaded
```

### 1.2 NVHPC SDK + CUDA

The DGX OS image typically includes the NVHPC SDK. If not:

```bash
sudo apt install -y nvhpc-25-x   # or whatever current major version is in apt
# Add to ~/.bashrc:
export NVHPC_ROOT=/opt/nvidia/hpc_sdk/Linux_aarch64/<version>
export PATH=$NVHPC_ROOT/compilers/bin:$PATH
export LD_LIBRARY_PATH=$NVHPC_ROOT/compilers/lib:$LD_LIBRARY_PATH
nvcc --version
mpicxx --version       # NVHPC ships its own MPI
```

### 1.3 Quantum ESPRESSO with CUDA

Build from source; the apt package is CPU-only.

```bash
sudo apt install -y libfftw3-dev libopenblas-dev
git clone https://gitlab.com/QEF/q-e.git
cd q-e
git checkout qe-7.3   # or whatever release the QE team currently recommends for GPU

./configure \
  --enable-openmp \
  --with-cuda=$NVHPC_ROOT/cuda \
  --with-cuda-cc=120 \
  --with-cuda-runtime=12.6
make -j$(nproc) pw

# Optional but recommended for this project:
make -j$(nproc) hp neb

# Add to PATH:
export PATH=$(pwd)/bin:$PATH
which pw.x
pw.x --version | head -3
```

Verify it actually uses the GPU on a one-atom Cu bulk SCF:

```bash
copper-oxide-dft bulk-cu --out /tmp/smoke/pw.in --ecutwfc 60
(cd /tmp/smoke && OMP_NUM_THREADS=4 mpirun -n 1 pw.x -in pw.in > pw.out)
grep -i "gpu" /tmp/smoke/pw.out | head
# Expect: "GPU acceleration is enabled" or similar.
```

If you see GPU lines, you're done. If not, the configure step picked up
the CPU path — re-do `./configure` with `--with-cuda=...` explicit.

### 1.4 This package

```bash
git clone <your-fork>/copper-oxide-dft.git
cd copper-oxide-dft
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
pytest -q   # expect 165 passed
```

### 1.5 Pseudopotentials

You need **Cu, O, and H** PseudoDojo PBE PAW pseudopotentials.

```bash
mkdir -p ~/pseudos
# Download Cu.upf, O.upf, H.upf from http://www.pseudo-dojo.org/
# (PBE, scalar-relativistic, standard accuracy, PAW)
mv ~/Downloads/{Cu,O,H}.upf ~/pseudos/
export CUOXDFT_PSEUDO_DIR=~/pseudos
echo 'export CUOXDFT_PSEUDO_DIR=~/pseudos' >> ~/.bashrc
```

### 1.6 A runner script (optional but useful)

To avoid retyping the `mpirun` line:

```bash
cat > ~/bin/qe-run <<'EOF'
#!/usr/bin/env bash
# qe-run <directory> — run pw.x on a pw.in inside <directory>
set -euo pipefail
cd "$1"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
mpirun -n "${QE_NRANKS:-1}" pw.x -in pw.in > pw.out
EOF
chmod +x ~/bin/qe-run
```

Then any `qe-run runs/<somewhere>` will run the calculation in that
directory and capture `pw.out` next to `pw.in`.

---

## 2. Phase 1 — Bulk Cu convergence

This is identical to the [Ubuntu walkthrough](local-workstation.md#phase-1--bulk-cu).
You need it because every subsequent phase needs converged `ecutwfc` and
k-point density for the Cu valence states.

### 2.1 Single SCF as a sanity check

```bash
copper-oxide-dft bulk-cu --out runs/bulk_cu/pw.in
copper-oxide-dft inspect runs/bulk_cu/pw.in
qe-run runs/bulk_cu
copper-oxide-dft parse runs/bulk_cu/pw.out
```

`parse` should print `done=True` and a total energy. On GB10 with GPU
acceleration this finishes in seconds.

### 2.2 Convergence sweep

```bash
copper-oxide-dft sweep --param ecutwfc --values 40,60,80,100 --out runs/conv_ecutwfc
for d in runs/conv_ecutwfc/*/; do qe-run "$d"; done

copper-oxide-dft sweep-analyze runs/conv_ecutwfc \
  --threshold-mev 1 \
  --png runs/conv_ecutwfc/convergence.png
```

Repeat for `kpts` (try 6, 8, 10, 12) and `degauss` (0.01, 0.02, 0.03).
Record the converged triplet in [ground_truths.md](ground_truths.md);
every other phase reuses these.

### 2.3 Lattice parameter (vc-relax)

```bash
copper-oxide-dft bulk-cu --out runs/bulk_cu_vc/pw.in \
  --calculation vc-relax --ecutwfc <converged>
qe-run runs/bulk_cu_vc
grep -A 4 "CELL_PARAMETERS" runs/bulk_cu_vc/pw.out | tail -4
```

Relaxed `a` must be within 0.5 % of 3.615 Å. If not, your `ecutwfc` was
not actually converged — back to 2.2.

---

## 3. Phase 2 — Bulk CuO + Hubbard U

For your scientific question we only need CuO (the cathodic regime
shouldn't see Cu₂O), but it costs little to do Cu₂O too and you'll want
it for context.

### 3.1 Generate both oxides

```bash
copper-oxide-dft make-pourbaix-inputs runs/oxides \
  --ecutwfc <converged-from-phase-1> \
  --hubbard-u 4.0
# This writes bulk_cu, bulk_cu2o, bulk_cuo, mol_h2, mol_h2o
copper-oxide-dft inspect runs/oxides/bulk_cuo/pw.in
```

Look for `starting_magnetization(1)` and `starting_magnetization(2)` on
the two Cu sub-species in the printed input — AFM CuO needs both, and
the writer has burned us once before (see [ground_truths.md](ground_truths.md)
2026-05-14: AFM CuO species splitting).

### 3.2 Run the AFM CuO SCF

```bash
qe-run runs/oxides/bulk_cuo
copper-oxide-dft parse runs/oxides/bulk_cuo/pw.out
```

Sanity:
- `job_done=True`
- `total_magnetization_bohr` near 0 (AFM cancels globally; per-site
  moments are large but the unit cell sums to zero).
- Band gap in the output around 1.2–1.7 eV (experimental range).

If the gap is < 0.5 eV or `total_magnetization_bohr` is very nonzero,
the AFM ordering didn't survive and you're in a wrong local minimum.
Re-check the per-atom `starting_magnetization` lines in `pw.in` and try
alternate moment patterns (e.g. AFM-I vs AFM-II) via
`atoms.set_initial_magnetic_moments(...)` before the writer call.

### 3.3 Hubbard U sweep (optional but defensible)

If you want to claim a U value rather than cite the literature:

```bash
python -c "
from copper_oxide_dft.convergence import sweep_convergence
from copper_oxide_dft.structure_builder import build_bulk_cuo
sweep_convergence(
    build_bulk_cuo(),
    out_root='runs/u_sweep_cuo',
    pseudopotentials={'Cu': 'Cu.upf', 'O': 'O.upf'},
    param='hubbard_u',
    values=[0.0, 2.0, 4.0, 6.0, 8.0],
)
"
for d in runs/u_sweep_cuo/*/; do qe-run "$d"; done

# Compare band gap to 1.2-1.7 eV experimental range; lock in your U.
```

Update [ground_truths.md](ground_truths.md) with the U you chose and why.

---

## 4. Phase 3 — Cu(111) + O adsorbates (the proxy for "CuO on Cu")

This is the most expensive prototype step on GB10. A 4-layer × 3×3
Cu(111) slab is 36 Cu atoms; adding a few O brings you to ~40-atom
spin-polarized DFT+U with k-point sampling that's still tractable
overnight.

### 4.1 Clean slab

```python
python -c "
from copper_oxide_dft.structure_builder import build_cu111_slab
from copper_oxide_dft.qe_input import write_pw_input

slab = build_cu111_slab(layers=4, supercell=(3, 3), vacuum_ang=20.0)
write_pw_input(
    slab,
    out_path='runs/cu111_clean/pw.in',
    pseudopotentials={'Cu': 'Cu.upf'},
    calculation='relax',
    kpts=(6, 6, 1),
    ecutwfc=<converged>,
)
"
copper-oxide-dft inspect runs/cu111_clean/pw.in   # 4 layers of 9 Cu
qe-run runs/cu111_clean
```

`kpts=(6,6,1)` is the slab convention: `kz=1` because sampling
perpendicular to a slab wastes work.

### 4.2 O-covered slabs at 1/4, 1/2, 3/4, 1 ML

A small Python driver since this isn't a CLI command yet:

```python
python -c "
from copper_oxide_dft.qe_input import write_pw_input, spin_and_hubbard_overrides
from copper_oxide_dft.structure_builder import build_cu111_slab, add_oxygen_adsorbates

ECUTWFC = <converged>          # from Phase 1
U_CU = 4.0                     # or your Phase 2 result

for coverage in (1/4, 1/2, 3/4, 1.0):
    slab = build_cu111_slab(layers=4, supercell=(3, 3), vacuum_ang=20.0)
    covered = add_oxygen_adsorbates(slab, coverage_ml=coverage, site='fcc')
    label = f'{int(coverage * 100):03d}'
    write_pw_input(
        covered,
        out_path=f'runs/cu111_O_{label}ML/pw.in',
        pseudopotentials={'Cu': 'Cu.upf', 'O': 'O.upf'},
        calculation='relax',
        kpts=(6, 6, 1),
        ecutwfc=ECUTWFC,
        extra_input_data=spin_and_hubbard_overrides(
            covered, nspin=2, hubbard_u={'Cu': U_CU}
        ),
    )
"

for d in runs/cu111_O_*/; do
    copper-oxide-dft inspect "$d/pw.in"   # eyeball the geometry
done
```

Then run them. They're overnight calculations per coverage on GB10
unless you have unusually fast convergence; expect 4–8 hours each.

```bash
for d in runs/cu111_O_*/; do qe-run "$d"; done
```

### 4.3 Surface energies and adsorption energies (in vacuum)

After the runs finish, compute the differential O adsorption energy for
each coverage:

```python
python -c "
from copper_oxide_dft.parse import parse_pw_output

E_clean = parse_pw_output('runs/cu111_clean/pw.out').total_energy_ev
E_O2_per_atom = parse_pw_output('runs/oxides/bulk_cuo/pw.out').total_energy_ev  # rough
# (better: run a true O2 reference; build_reference_o2 handles the triplet)

for label, coverage, n_o in [('025', 0.25, 2),  # 2 atoms in a 3x3 = 2/9 ≈ 1/4 ML
                              ('050', 0.5, 4),
                              ('075', 0.75, 7),
                              ('100', 1.0, 9)]:
    E_cov = parse_pw_output(f'runs/cu111_O_{label}ML/pw.out').total_energy_ev
    dE = (E_cov - E_clean - n_o * (E_O2_per_atom / 2)) / n_o
    print(f'{label} ML: ΔE_ads(O) = {dE:+.3f} eV / O atom')
"
```

Sanity: dilute coverage should give the most negative (strongest)
adsorption energy; the literature value for 1/4 ML O on Cu(111) is
around −4.5 to −5 eV vs. ½O₂. If your number is way off, suspect (in
order) wrong O₂ reference, wrong spin treatment, unconverged cutoffs.

These vacuum surface energetics are the **input** to Phase 7 — at
constant potential, ΔE(coverage) translates to ΔG(U, coverage), and the
lowest-G state is what the surface relaxes toward.

---

## 5. Phase 5 — Non-aqueous implicit solvation (validation on DGX Spark)

If the Environ-patched QE build is available on your DGX Spark
installation, you can include implicit solvent at the implicit (mean-field)
level. If not — and this is likely on a fresh GB10 install — Environ is
an optional refinement and you can move to Phase 7 in vacuum. Both
paths are documented below.

### 5.1 Generate the environ.in for THF

The Environ writer in this repo defaults to water (ε = 78.36). Override
for THF:

```python
python -c "
from copper_oxide_dft.environ import write_environ_input

# THF: ε = 7.52 at 298 K (CRC handbook).
# 1 % EtOH adds <0.1 to ε; ignored at the implicit-solvent level.
write_environ_input(
    'runs/cu111_O_025ML/environ.in',
    environ_type='input',          # NOT 'water' — we override the value
    static_permittivity=7.52,
)
"
cat runs/cu111_O_025ML/environ.in
```

> **Note**: passing `environ_type='input'` is the Environ convention for
> "use the supplied numbers rather than a named preset." If you stay
> with `environ_type='water'` while overriding `static_permittivity`,
> Environ silently keeps its built-in water parameters and you've lied
> to it about the solvent. Verify against your Environ version's docs.

> **Why THF matters less than water did**: with ε = 7.52, the implicit
> solvent screens a charged surface roughly 10× less than water does.
> The energetic shifts between vacuum and implicit-THF will be small —
> the *order* of stable coverages at U = −0.8 V is unlikely to change.
> Implicit solvation here is more about being correct than about
> changing the answer. If you want the EtOH proton donor to actually
> matter (e.g. for PCET steps), you need explicit EtOH molecules
> (Phase 6 territory), not implicit solvent.

### 5.2 Check whether your QE is Environ-patched

```bash
pw.x --help 2>&1 | grep -i environ
# Or: try to run a vacuum case with a benign environ.in next to pw.in.
# Patched pw.x parses both files; stock pw.x ignores environ.in silently.
```

If unpatched, building Environ-patched QE on ARM/CUDA from scratch is a
half-day exercise; postpone until Frontier or do it once and document.

### 5.3 Re-run one O-covered slab with implicit solvent

```bash
cp runs/cu111_O_025ML/pw.in     runs/cu111_O_025ML_solv/pw.in
cp runs/cu111_O_025ML/environ.in runs/cu111_O_025ML_solv/environ.in   # if you wrote one
qe-run runs/cu111_O_025ML_solv
diff <(grep "^!    total energy" runs/cu111_O_025ML/pw.out | tail -1) \
     <(grep "^!    total energy" runs/cu111_O_025ML_solv/pw.out | tail -1)
```

Expect the solvent run to shift the total energy by a fraction of an eV;
the sign and magnitude depend on how polar the surface is.

---

## 6. Phase 7 — ESM-FCP at U = −0.8 V

This is the answer to your scientific question: at fixed U = −0.8 V vs.
your chosen reference electrode, how does the surface evolve?

The expensive Phase-7 calculations belong on Frontier (Section 7).
Validate the input file *generation* and a short SCF on DGX Spark
first.

### 6.1 Verify QE on DGX Spark accepts `lfcp`

```bash
python -c "
from copper_oxide_dft.qe_input import (
    fcp_overrides_for_potential, spin_and_hubbard_overrides, write_pw_input,
)
from copper_oxide_dft.structure_builder import build_cu111_slab

slab = build_cu111_slab(layers=4, supercell=(2, 2), vacuum_ang=20.0)
fcp = fcp_overrides_for_potential(-0.8, she_absolute_v=4.64)   # Ag/AgCl
spin = spin_and_hubbard_overrides(slab, nspin=1)
merged = {}
for src in (fcp, spin):
    for nm, entries in src.items():
        merged.setdefault(nm, {}).update(entries)
write_pw_input(
    slab, out_path='runs/cu111_fcp_smoke/pw.in',
    pseudopotentials={'Cu': 'Cu.upf'},
    calculation='scf', kpts=(6, 6, 1),
    extra_input_data=merged,
)
"
grep -E 'lfcp|assume_isolated|esm_bc|fcp_mu' runs/cu111_fcp_smoke/pw.in
# Expect lfcp=.true., assume_isolated='esm', esm_bc='bc2', fcp_mu ≈ -0.282 Ry
```

Run a single SCF iteration to confirm parsing:

```bash
qe-run runs/cu111_fcp_smoke
head -50 runs/cu111_fcp_smoke/pw.out | grep -i -E 'esm|fcp'
```

If you see `Effective Screening Medium method` and `Fictitious Charge`
lines, QE accepted ESM-FCP. If the run errors at `lfcp` or
`assume_isolated`, your QE build is too old — rebuild from a recent
release (≥ qe-7.2).

### 6.2 Generate the full coverage × potential matrix (input only)

You need ESM-FCP-converged runs at U = −0.8 V for at least clean Cu(111)
plus the O-coverage series from Phase 3. The same Python pattern as
Section 4.2:

```python
python -c "
from copper_oxide_dft.qe_input import (
    fcp_overrides_for_potential, spin_and_hubbard_overrides, write_pw_input,
)
from copper_oxide_dft.structure_builder import build_cu111_slab, add_oxygen_adsorbates

ECUTWFC = <converged>
U_CU = 4.0
U_TARGET = -0.8
V_ABS = 4.64   # Ag/AgCl (= SHE 4.44 + Ag/AgCl-vs-SHE 0.197).

bases = [
    ('clean', lambda s: s),
    ('O_025ML', lambda s: add_oxygen_adsorbates(s, coverage_ml=0.25, site='fcc')),
    ('O_050ML', lambda s: add_oxygen_adsorbates(s, coverage_ml=0.5, site='fcc')),
    ('O_075ML', lambda s: add_oxygen_adsorbates(s, coverage_ml=0.75, site='fcc')),
    ('O_100ML', lambda s: add_oxygen_adsorbates(s, coverage_ml=1.0, site='fcc')),
]

for label, modify in bases:
    slab = modify(build_cu111_slab(layers=4, supercell=(3, 3), vacuum_ang=20.0))
    fcp = fcp_overrides_for_potential(U_TARGET, she_absolute_v=V_ABS)
    spin = spin_and_hubbard_overrides(slab, nspin=2, hubbard_u={'Cu': U_CU})
    merged = {}
    for src in (fcp, spin):
        for nm, entries in src.items():
            merged.setdefault(nm, {}).update(entries)
    write_pw_input(
        slab,
        out_path=f'runs/fcp_minus0p8V/{label}/pw.in',
        pseudopotentials={'Cu': 'Cu.upf', 'O': 'O.upf'},
        calculation='relax', kpts=(6, 6, 1),
        ecutwfc=ECUTWFC,
        extra_input_data=merged,
    )
"
for d in runs/fcp_minus0p8V/*/; do copper-oxide-dft inspect "$d/pw.in"; done
```

### 6.3 Don't run the full matrix on DGX Spark

A 36-Cu + adsorbate slab with ESM-FCP + DFT+U + spin polarization will
take many hours per relaxation step on GB10. Validate inputs locally;
run on Frontier.

---

## 7. Ship to Frontier

Once Phases 1–3 are converged and the Phase-7 inputs look right:

```bash
# 1. Wrap each pw.in with a SLURM script targeting Frontier.
copper-oxide-dft make-slurm runs/fcp_minus0p8V \
  --account <YOUR_PROJECT> \
  --walltime 2:00:00 \
  --qe-module quantum-espresso/<version>-gpu

# 2. Ship.
rsync -av runs/fcp_minus0p8V/ frontier:scratch/fcp_minus0p8V/
rsync -av ~/pseudos/ frontier:scratch/pseudos/          # if first time

# 3. Submit on Frontier.
ssh frontier
echo "export CUOXDFT_PSEUDO_DIR=/lustre/.../pseudos" >> ~/.bashrc
for d in /lustre/.../scratch/fcp_minus0p8V/*/; do
    (cd "$d" && sbatch submit.sh)
done

# 4. Pull results, parse.
rsync -av frontier:/lustre/.../scratch/fcp_minus0p8V/ runs/fcp_minus0p8V/
for d in runs/fcp_minus0p8V/*/; do
    copper-oxide-dft parse "$d/pw.out"
done
```

The Frontier-side defaults in `SlurmConfig.for_frontier` already encode
the cluster's MPI/GPU conventions (8 GCDs per node, GPU-aware MPICH,
closest-binding) — see [ground_truths.md](ground_truths.md): Frontier
SLURM conventions.

### 7.1 Free-energy ranking at U = −0.8 V

At each (coverage, U) point you have:

- DFT total energy `E_DFT(coverage; U)` from the FCP-converged run
- Total electron count `N_e(coverage; U)` from the FCP loop

The grand-canonical free energy is `Ω = E_DFT − μ_e · N_e`, with
`μ_e = -(V_abs + U)`. The lowest-Ω coverage at U = −0.8 V is the
predicted surface state.

This step is not yet in [pourbaix.py](../src/copper_oxide_dft/pourbaix.py)
(it's aqueous-only) so write it inline for now:

```python
python -c "
from copper_oxide_dft.parse import parse_pw_output
V_ABS = 4.64    # Ag/AgCl absolute
U = -0.8
mu_e = -(V_ABS + U)    # eV vs vacuum

# Parse each FCP-converged pw.out; QE prints the converged electron count
# in the FCP section. The parse module currently extracts total energy,
# Fermi energy, and magnetization; the FCP N_e extraction is TODO.
# For now, grep manually:
import re
for label in ('clean','O_025ML','O_050ML','O_075ML','O_100ML'):
    text = open(f'runs/fcp_minus0p8V/{label}/pw.out').read()
    e_dft = parse_pw_output(f'runs/fcp_minus0p8V/{label}/pw.out').total_energy_ev
    m = re.search(r'tot_charge\s*=\s*([-+0-9.eE]+)', text[-2000:])
    q = float(m.group(1)) if m else 0.0
    omega = e_dft - mu_e * (-q)   # N_e relative-to-neutral = -tot_charge
    print(f'{label:>10}  E={e_dft:+.4f} eV  q={q:+.3f}  Ω-ref={omega:+.4f} eV')
"
```

> **TODO**: `parse_fcp_output(...)` in
> [parse.py](../src/copper_oxide_dft/parse.py) that extracts the
> converged FCP electron count alongside the existing scalars, plus a
> `grand_canonical_free_energy` helper in `che.py` for the non-aqueous
> ranking. The current `che.py` is aqueous-Pourbaix-only; this is a
> small addition once the structure of the FCP output stabilises.

---

## What "good" looks like

By the end of this guide you have:

- ✅ A converged QE+CUDA build on DGX Spark
- ✅ Phase-1 converged `ecutwfc` / kpts / degauss for your Cu system
- ✅ A Hubbard-U value chosen and recorded in `ground_truths.md`
- ✅ Vacuum surface and adsorption energies for Cu(111) at 0, 1/4, 1/2, 3/4, 1 ML O
- ✅ A reference-electrode convention (Fc/Fc⁺ in MeCN) committed to `ground_truths.md`
- ✅ A validated ESM-FCP input file at U = −0.8 V
- ✅ Frontier-side production runs in flight or complete
- ✅ A free-energy ranking of clean / O-covered Cu(111) at U = −0.8 V

The result you're after is one of:

- **Cu metal wins everywhere.** The cathodic regime suppresses surface
  oxidation — consistent with the experimental Cu Pourbaix at U < 0 V.
  Story is "predicted from first principles."
- **A reconstruction wins.** Some O coverage or a relaxed Cu surface with
  cation displacement is lower in Ω than clean Cu(111). Interesting
  science; next step is to build the explicit reconstructed structure
  (Phase 8).
- **A coverage transition appears as U sweeps.** If you do U ∈ {−1.2,
  −0.8, −0.4} V instead of just −0.8 V, the crossover potential is
  experimentally measurable; worth doing as the second batch.

---

## Pitfalls specific to this workflow

- **"U vs Ag/AgCl is U vs SHE"** — it isn't. Off by +0.197 V (Ag/AgCl
  sat. KCl). Off by even more in non-aqueous, where Ag/AgCl is a
  pseudo-reference that drifts. **Lock `she_absolute_v=4.64` in every
  call site** and *never* mix references mid-project. If you later
  calibrate against Fc/Fc⁺, re-derive and update everywhere at once.
- **"THF screens like water does"** — it doesn't. ε(THF) = 7.52 vs
  ε(water) = 78.36. Implicit solvation in THF moves total energies by
  tens of meV, not hundreds. Don't expect the implicit-solvent
  correction to change which coverage wins — for that you'd need
  explicit EtOH.
- **"The EtOH proton donor doesn't matter for ESM-FCP"** — true at fixed
  U with no PCET. But if you later switch to mechanism analysis (e.g.
  "does -O at 1/4 ML get protonated and reduced at U = −0.8 V?") then
  the μ(H⁺ + e⁻) reference is set by EtOH ⇌ EtO⁻ + H⁺, *not* H₂O ⇌ OH⁻
  + H⁺. The CHE machinery in `che.py` is hard-coded to the water
  reservoir — adapting it to EtOH is a future task.
- **"CHE gives me the answer for free"** — only in aqueous. The Phase-4
  Pourbaix machinery here will appear to run on non-aqueous data and
  produce qualitatively meaningless diagrams. Skip Phase 4.
- **"Cu(111) + O adsorbates *is* CuO"** — it's a proxy. Cu(111) + 1 ML
  O is geometrically not CuO (CuO has 5-fold Cu coordination; the
  adsorbate model has 3-fold). The proxy is for ranking, not for
  structure prediction. The honest structure for "CuO on Cu" is the
  coincident-supercell builder that doesn't exist yet (Section 0.2).
- **"GB10 GPU is just like Frontier"** — Blackwell ≠ MI250X. Performance
  scaling, peak memory, MPI rank counts are all different. Don't extrapolate
  GB10 wall times to Frontier; benchmark on a small Frontier run first.
- **"ESM-FCP converged on iteration 1"** — suspicious. FCP should require
  several outer iterations to align the Fermi level. A 1-iteration
  "convergence" usually means `fcp_thr` is too loose.
- **Pseudopotential filename mismatch** — the CLI defaults assume
  `Cu.upf`, `O.upf`, `H.upf`. PseudoDojo downloads with longer names;
  rename or pass `--cu-pseudo`/`--o-pseudo`/`--h-pseudo` everywhere.

---

## Related docs

- [implementation-plan.md](implementation-plan.md) — full 9-phase
  roadmap (aqueous default).
- [local-workstation.md](local-workstation.md) — Ubuntu CPU walkthrough,
  covers Phases 1–8 with the aqueous defaults. Use as the cross-reference
  for any phase this guide is brief on.
- [ground_truths.md](ground_truths.md) — methodology decisions, AFM CuO
  gotchas, Frontier conventions. **Update with your non-aqueous reference
  electrode choice before you do anything else.**
- [project.md](project.md) — scope and dependencies.
