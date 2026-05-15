# Local Workstation Walkthrough (Ubuntu, CPU)

End-to-end local validation of the **full project** (Phases 1–8) before any Frontier hours are spent. The target hardware is an Ubuntu workstation with multi-core CPU; this guide intentionally **does not** use the Radeon RX 7900 XT GPUs for Quantum ESPRESSO (see [GPU notes](#a-note-on-amd-rx-7900-xt-gpus) at the end).

The point of running locally is to **validate the toolchain, the structures, and the analysis pipeline** before any of it touches Frontier hours. Bulk Cu and the bulk oxides are small enough that even one CPU thread converges in minutes; the slab + Phase 4 v2 work is slower but still tractable on a desktop overnight. Phases 5–8 require specialized QE builds (Environ patch, ESM/FCP, NEB) that the stock apt package may not include — for those, this guide focuses on **input-file verification** rather than full execution, with explicit notes on which extra builds are needed.

**Tip: keep an `inspect` window open**. `copper-oxide-dft inspect runs/<anything>/pw.in` summarises composition, cell, and layer-by-layer geometry, and is the cheapest way to catch a malformed structure before submitting an expensive job.

---

## What you can validate locally vs. what waits for Frontier

| Phase | Locally? | Why |
|---|---|---|
| 1 — Bulk Cu (vc-relax, convergence sweep) | ✅ Full | Trivial cell, seconds per SCF on CPU |
| 2 — Bulk Cu₂O + CuO + Hubbard U | ✅ Full | 6 / 8 atoms; CuO slower (AFM + U) but works |
| 3 — Cu(111) slab + adsorbates | ✅ Mostly | Use 2×2 supercell for smoke tests; structural verification is free |
| 4 — CHE Pourbaix (bulk and adsorbate) | ✅ Full | Pure post-processing — no QE re-run needed once Phase 1–3 energies are in hand |
| 5 — Environ implicit solvation | ⚠️ Input only | Requires Environ-patched QE; stock apt build does not have it |
| 6 — Explicit water layer | ✅ Structural only | Builder is pure Python; full DFT runs are Frontier-scale |
| 7 — ESM-FCP constant-potential | ⚠️ Input only | Stock QE has ESM but FCP support varies by version |
| 8 — NEB reconstruction barriers | ⚠️ Input only | `neb.x` is in the QE package but per-image runs are slow |

For the "Input only" rows, this walkthrough has you generate the input file and have QE parse it (using a tiny test calculation), but does not attempt to converge a real production run.

---

## 0. Prerequisites

- Ubuntu 22.04 LTS or newer.
- Python 3.10+.
- `git`, `build-essential`, `gfortran`, `mpich` or `openmpi`.

```bash
sudo apt update
sudo apt install -y build-essential gfortran python3.10-venv git
```

## 1. Install Quantum ESPRESSO (CPU)

```bash
sudo apt install -y quantum-espresso
which pw.x          # /usr/bin/pw.x
which neb.x         # /usr/bin/neb.x (needed for Phase 8)
which hp.x          # /usr/bin/hp.x (needed for Phase 2 U calibration)
pw.x --version | head -3
```

This is typically QE 6.7–7.0. If any of `pw.x` / `neb.x` / `hp.x` is missing, see [Building QE from source](#optional-building-qe-from-source).

## 2. Install this package

```bash
git clone <your-fork>/copper-oxide-dft.git
cd copper-oxide-dft
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
pytest -q   # should print "165 passed"
```

If 165 tests pass, the Python side is fully functional independent of whether QE itself works.

## 3. Pseudopotentials

Grab the **Cu, O, and H** pseudopotentials from [PseudoDojo](http://www.pseudo-dojo.org/). We need all three for Phases 2–8:

- Functional: **PBE**
- Type: **scalar-relativistic**
- Accuracy: **standard**
- Format: **PAW (UPF)**

```bash
mkdir -p ~/pseudos
mv ~/Downloads/Cu.upf ~/Downloads/O.upf ~/Downloads/H.upf ~/pseudos/
export CUOXDFT_PSEUDO_DIR=~/pseudos
echo 'export CUOXDFT_PSEUDO_DIR=~/pseudos' >> ~/.bashrc
```

The filename matters — the CLI defaults assume `Cu.upf`, `O.upf`, `H.upf`. If PseudoDojo gives you longer filenames like `Cu.UPF.PBE.standard.upf`, either rename to the short form or pass `--cu-pseudo` / `--o-pseudo` / `--h-pseudo` to every CLI command.

---

## Phase 1 — Bulk Cu

### 1a. Single input + inspect

```bash
copper-oxide-dft bulk-cu --out runs/bulk_cu/pw.in
copper-oxide-dft inspect runs/bulk_cu/pw.in
```

Expected:

```
File:        runs/bulk_cu/pw.in
Composition: Cu (1 atoms)
Volume:      11.8104 A^3
Cell vectors (A):
  a: (   0.0000    1.8075    1.8075)  |a| = 2.5562
  ...
Layers grouped by z (tol=0.1 A):
  [ 0] z =   0.0000 A  thickness = 0.0000 A  Cux1  (1 atoms)
```

Sanity: `|a|³ × √2 ≈ 3.615³ ≈ 47.2 Å³`; the primitive cell holds 1 of 4 conventional-cell atoms, so the volume here should be `47.2 / 4 ≈ 11.8 Å³`. ✓

### 1b. Run the SCF locally

```bash
cd runs/bulk_cu
pw.x -in pw.in > pw.out
# Multi-threaded:
# mpirun -n 4 pw.x -in pw.in > pw.out
cd -

copper-oxide-dft parse runs/bulk_cu/pw.out   # done=True
```

### 1c. Convergence sweep + analysis

```bash
copper-oxide-dft sweep \
  --param ecutwfc --values 40,60,80 \
  --out runs/conv_ecutwfc

for d in runs/conv_ecutwfc/*/; do
    (cd "$d" && pw.x -in pw.in > pw.out)
done

copper-oxide-dft sweep-analyze runs/conv_ecutwfc \
  --threshold-mev 1 \
  --png runs/conv_ecutwfc/convergence.png
```

`sweep-analyze` walks the tree, parses each output, picks the smallest `ecutwfc` that converges total energy/atom to within 1 meV, and saves a plot. If it reports "NONE within threshold" the sweep needs to be extended upward — that's a feature, not a bug.

Try the other two parameters too:

```bash
copper-oxide-dft sweep --param kpts --values 6,8,10 --out runs/conv_kpts
copper-oxide-dft sweep --param degauss --values 0.01,0.02,0.03 --out runs/conv_degauss
```

### 1d. Lattice-parameter validation (vc-relax)

```bash
copper-oxide-dft bulk-cu \
  --out runs/bulk_cu_vc/pw.in \
  --calculation vc-relax \
  --ecutwfc 80
cd runs/bulk_cu_vc && pw.x -in pw.in > pw.out && cd -
grep -A 4 "CELL_PARAMETERS" runs/bulk_cu_vc/pw.out | tail -4
```

The relaxed `a` should be within ~0.5 % of 3.615 Å. If it isn't, **stop and debug** — every subsequent phase inherits this.

---

## Phase 2 — Bulk Cu₂O + CuO + Hubbard U

### 2a. Cu₂O (non-magnetic, with U)

The Phase 4 input bundler writes a Cu₂O vc-relax input by default:

```bash
copper-oxide-dft make-pourbaix-inputs runs/phase4 --pseudo-dir ~/pseudos
copper-oxide-dft inspect runs/phase4/bulk_cu2o/pw.in   # 6 atoms, cubic, Cu+O
```

Run it:

```bash
cd runs/phase4/bulk_cu2o && pw.x -in pw.in > pw.out && cd -
copper-oxide-dft parse runs/phase4/bulk_cu2o/pw.out
```

This takes a few minutes single-threaded; with `mpirun -n 4` it drops to ~1 minute.

### 2b. CuO (antiferromagnetic, with U)

```bash
copper-oxide-dft inspect runs/phase4/bulk_cuo/pw.in   # 8 atoms, monoclinic
cd runs/phase4/bulk_cuo && mpirun -n 4 pw.x -in pw.in > pw.out && cd -
copper-oxide-dft parse runs/phase4/bulk_cuo/pw.out
```

CuO is the trickiest of the bulks: AFM ordering + DFT+U + low-symmetry cell. The parsed output should report a finite `total_magnetization_bohr` (close to 0 for the AFM ground state — the total cancels but per-site moments are large). If `total_magnetization_bohr` is `None` or the SCF didn't converge, the AFM starting moments are probably wrong; check that `inspect` shows `starting_magnetization` in the input. The walkthrough in `tests/test_qe_input.py::test_spin_and_hubbard_overrides_afm_cuo_splits_cu_into_two_species` documents the species-splitting quirk that bit us in development.

### 2c. Hubbard-U calibration sweep (optional)

To defend a U value rather than using the literature default (4 eV):

```bash
# Sweep U on Cu₂O at fixed cutoffs (this is the easier of the two).
python -c "
from copper_oxide_dft.convergence import sweep_convergence
from copper_oxide_dft.structure_builder import build_bulk_cu2o
sweep_convergence(
    build_bulk_cu2o(),
    out_root='runs/u_sweep_cu2o',
    pseudopotentials={'Cu': 'Cu.upf', 'O': 'O.upf'},
    param='hubbard_u',
    values=[0.0, 2.0, 4.0, 6.0, 8.0],
)
"
for d in runs/u_sweep_cu2o/*/; do
    (cd "$d" && mpirun -n 4 pw.x -in pw.in > pw.out)
done
```

Then visually compare each output's band gap to experiment (Cu₂O: 2.17 eV). The Phase 1+2 success criterion is "within ~30 % of experiment, lattice parameters within ~2 %" — record the U you pick and add it to `docs/ground_truths.md`.

### 2d. `hp.x` linear-response U (optional sanity check)

For self-consistent U via density-functional perturbation theory:

```python
python -c "
from copper_oxide_dft.qe_input import write_hp_input
write_hp_input('runs/phase4/bulk_cu2o/hp.in', prefix='bulk_cu2o', nq=(2,2,2))
"
cd runs/phase4/bulk_cu2o && hp.x -in hp.in > hp.out && cd -
```

The output should converge on a U value close to your chosen literature pick. Discrepancies > 1 eV mean either the SCF reference wasn't fully converged or the q-grid is too coarse — bump `nq` to (3,3,3) and retry.

---

## Phase 3 — Surfaces and adsorbates

### 3a. Cu(111) slab structural verification (fast)

The structural side requires no DFT — `inspect` does it all:

```bash
python -c "
from copper_oxide_dft.structure_builder import build_cu111_slab
from copper_oxide_dft.qe_input import write_pw_input
slab = build_cu111_slab(layers=4, supercell=(3, 3))
write_pw_input(slab, out_path='runs/cu111/pw.in',
               pseudopotentials={'Cu': 'Cu.upf'}, calculation='relax')
"
copper-oxide-dft inspect runs/cu111/pw.in
```

You should see 4 layers of 9 Cu atoms each, with ~2.1 Å spacing and 15 Å of vacuum at the top. If any layer has the wrong atom count or the spacing is wildly off, the slab builder is misaligned for your ASE version — file an issue.

### 3b. Cu(111) slab SCF (slow but tractable)

A 36-atom slab is borderline for CPU. Use a 2×2 supercell (16 atoms) for the smoke test:

```bash
python -c "
from copper_oxide_dft.structure_builder import build_cu111_slab
from copper_oxide_dft.qe_input import write_pw_input
slab = build_cu111_slab(layers=4, supercell=(2, 2))
write_pw_input(slab, out_path='runs/cu111_2x2/pw.in',
               pseudopotentials={'Cu': 'Cu.upf'}, calculation='scf',
               kpts=(6, 6, 1))   # kz=1 for slabs
"
cd runs/cu111_2x2 && mpirun -n 4 pw.x -in pw.in > pw.out && cd -
copper-oxide-dft parse runs/cu111_2x2/pw.out
```

The kpts trick (`kz=1`) is mandatory for slabs — sampling perpendicular to the surface wastes work.

### 3c. Surface-energy validation

Compute the Cu(111) surface energy and compare to the literature (~0.08 eV/Å² ≈ 1.3 J/m²):

```python
python -c "
from copper_oxide_dft.parse import parse_pw_output
from copper_oxide_dft.structure_builder import surface_energy_ev_per_a2
import numpy as np

slab = parse_pw_output('runs/cu111_2x2/pw.out').total_energy_ev
bulk_per_atom = parse_pw_output('runs/bulk_cu/pw.out').total_energy_ev

# 4 layers × 4 atoms = 16; in-plane area = √3/2 × a² × (2×2) for fcc(111).
a = 3.615
area = (np.sqrt(3) / 2.0) * (a / np.sqrt(2))**2 * 4   # rough
gamma = surface_energy_ev_per_a2(slab, bulk_per_atom, n_atoms_in_slab=16,
                                  surface_area_ang2=area, n_surfaces=2)
print(f'γ = {gamma:.4f} eV/Å² = {gamma * 16.022:.3f} J/m²')
"
```

Expect something in the 0.05–0.10 eV/Å² range; small slab + coarse k-grid will overshoot.

### 3d. Add adsorbates

```python
python -c "
from copper_oxide_dft.structure_builder import build_cu111_slab, add_oxygen_adsorbates
from copper_oxide_dft.qe_input import (
    write_pw_input, spin_and_hubbard_overrides,
)
slab = build_cu111_slab(layers=4, supercell=(2, 2))
covered = add_oxygen_adsorbates(slab, coverage_ml=0.25, site='fcc')
write_pw_input(
    covered, out_path='runs/cu111_O_quarter_ml/pw.in',
    pseudopotentials={'Cu': 'Cu.upf', 'O': 'O.upf'},
    calculation='relax', kpts=(6, 6, 1),
    extra_input_data=spin_and_hubbard_overrides(covered, nspin=2),
)
"
copper-oxide-dft inspect runs/cu111_O_quarter_ml/pw.in
```

You should see one O atom above the slab on top of an fcc-hollow site. Spin polarization is enabled (any O-containing slab needs `nspin=2`). Then run the SCF as in 3b.

---

## Phase 4 — Pourbaix end-to-end

This is the headline question for your project — and the post-processing requires zero additional QE runs once Phases 1–3 land.

### 4a. Bulk-phase Pourbaix from real local energies

If you completed Phase 1d + Phase 2a/2b (or just used Phase 4's `make-pourbaix-inputs` and ran all five jobs), you have a real local DFT energy bundle:

```bash
# Make sure all five outputs exist.
ls runs/phase4/{bulk_cu,bulk_cu2o,bulk_cuo,mol_h2,mol_h2o}/pw.out
```

If `mol_h2/pw.out` and `mol_h2o/pw.out` are missing, run them now — they're trivial gas-phase SCFs:

```bash
cd runs/phase4/mol_h2 && pw.x -in pw.in > pw.out && cd -
cd runs/phase4/mol_h2o && pw.x -in pw.in > pw.out && cd -
```

Bridge to the Pourbaix CLI:

```bash
copper-oxide-dft aggregate-pourbaix-energies runs/phase4 \
  --out runs/phase4/energies.json

copper-oxide-dft pourbaix \
  --u -0.4 --ph 7 \
  --energies runs/phase4/energies.json \
  --png runs/phase4/pourbaix.png \
  --json runs/phase4/diagram.json
```

This is the moment of truth: your local DFT energies, fed through the CHE machinery, should produce a diagram that qualitatively matches the experimental Cu Pourbaix (Cu metal at low U, Cu₂O wedge near (0, 7), CuO at high U/high pH). If the topology is right, you can hand the same workflow to Frontier with confidence.

If the topology is *wrong* (e.g. CuO stable at -0.4 V/pH 7), the most likely culprit is wrong U value or unconverged cutoffs — go back to Phase 1c and tighten before scaling up.

### 4b. Adsorbate Pourbaix on Cu(111)

To build a coverage Pourbaix at the surface level, you need DFT energies for the clean Cu(111) slab and at least one covered version. Hand-build the inputs to `adsorbate_phase_diagram`:

```python
python -c "
from copper_oxide_dft.che import AdsorbateState, ReferenceEnergetics
from copper_oxide_dft.parse import parse_pw_output
from copper_oxide_dft.pourbaix import adsorbate_phase_diagram, plot_diagram
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

clean_e = parse_pw_output('runs/cu111_2x2/pw.out').total_energy_ev
o_e = parse_pw_output('runs/cu111_O_quarter_ml/pw.out').total_energy_ev
h2_e = parse_pw_output('runs/phase4/mol_h2/pw.out').total_energy_ev
h2o_e = parse_pw_output('runs/phase4/mol_h2o/pw.out').total_energy_ev

refs = ReferenceEnergetics(e_h2_ev=h2_e, e_h2o_ev=h2o_e)
clean = AdsorbateState(name='clean Cu(111)', n_adsorbed_o=0,
                       n_adsorbed_oh=0, e_dft_ev=clean_e)
o25 = AdsorbateState(name='1/4 ML O', n_adsorbed_o=1,
                     n_adsorbed_oh=0, e_dft_ev=o_e)

diagram = adsorbate_phase_diagram([clean, o25], clean, refs)
print('Stable at -0.4 V, pH 7:', diagram.stable_phase_at(-0.4, 7.0))

ax = plot_diagram(diagram, mark_point=(-0.4, 7.0))
ax.figure.savefig('runs/cu111_adsorbate_pourbaix.png', dpi=150,
                  bbox_inches='tight')
"
```

For a real surface Pourbaix you'd want multiple coverages (1/4, 1/2, 3/4, 1 ML) and OH as well as O — same machinery, more `AdsorbateState` entries.

---

## Phase 5 — Environ implicit solvation (input-only locally)

> **The stock apt QE package does NOT include the Environ patch.** You can verify input-file generation locally; running implicit-solvation calculations requires either building Environ-patched QE locally (see below) or doing this phase on Frontier (if your group has a patched build there).

### 5a. Verify input generation

```python
python -c "
from copper_oxide_dft.environ import write_environ_input
write_environ_input('runs/environ_smoke/environ.in', environ_type='water')
"
cat runs/environ_smoke/environ.in
```

You should see `&ENVIRON`, `&BOUNDARY`, `&ELECTROSTATIC` namelists. For a real run, place an `environ.in` next to a normal `pw.in` and invoke the Environ-built `pw.x` (which reads both files automatically).

### 5b. (Optional) Build Environ-patched QE locally

The Environ project provides build scripts that patch a QE source tree. See <https://environ.readthedocs.io/en/latest/install/install.html>. Roughly:

```bash
git clone https://gitlab.com/qe-environ/environ.git
git clone https://gitlab.com/QEF/q-e.git
cd q-e && git checkout qe-7.2   # match Environ's required QE version
cd ../environ && ./configure --prefix=$(pwd)/install
make -j$(nproc) compile
# Then build QE against the patched source per Environ's instructions.
```

Plan a half-day for this on a workstation if you've never built QE before.

---

## Phase 6 — Explicit water layer (structural-only locally)

```python
python -c "
from copper_oxide_dft.structure_builder import (
    build_cu111_slab, add_oxygen_adsorbates, add_explicit_water_layer,
)
from copper_oxide_dft.qe_input import (
    write_pw_input, spin_and_hubbard_overrides,
)
slab = build_cu111_slab(layers=4, supercell=(3, 3), vacuum_ang=20.0)
covered = add_oxygen_adsorbates(slab, coverage_ml=1/9, site='fcc')
hydrated = add_explicit_water_layer(covered, n_waters=12,
                                     height_ang=2.5, seed=42)
write_pw_input(
    hydrated, out_path='runs/cu111_O_water/pw.in',
    pseudopotentials={'Cu': 'Cu.upf', 'O': 'O.upf', 'H': 'H.upf'},
    calculation='relax', kpts=(4, 4, 1),
    extra_input_data=spin_and_hubbard_overrides(hydrated, nspin=2),
)
"
copper-oxide-dft inspect runs/cu111_O_water/pw.in
```

You should see four Cu layers, then an O adsorbate, then a stack of waters. Don't attempt the SCF on a CPU box — a 36-Cu + 12-water cell with spin polarization is multi-hour territory. The point is to verify the geometry looks sane before sending it to Frontier.

---

## Phase 7 — ESM-FCP constant-potential (input-only locally)

> Stock QE 7.x has ESM (Effective Screening Medium) but FCP support varies between versions. Verify with `pw.x` whether `lfcp` is a recognized control-namelist key in your build; if not, build from source on the `qe-7.3.1` tag or newer.

### 7a. Generate an ESM-FCP input

```python
python -c "
from copper_oxide_dft.qe_input import (
    fcp_overrides_for_potential, spin_and_hubbard_overrides, write_pw_input,
)
from copper_oxide_dft.structure_builder import build_cu111_slab

slab = build_cu111_slab(layers=4, supercell=(2, 2), vacuum_ang=20.0)
fcp = fcp_overrides_for_potential(-0.4)            # target -0.4 V vs SHE
spin = spin_and_hubbard_overrides(slab, nspin=1)   # clean Cu(111): no spin
merged = {}
for src in (fcp, spin):
    for nm, entries in src.items():
        merged.setdefault(nm, {}).update(entries)
write_pw_input(
    slab, out_path='runs/cu111_fcp/pw.in',
    pseudopotentials={'Cu': 'Cu.upf'},
    calculation='scf', kpts=(6, 6, 1),
    extra_input_data=merged,
)
"
copper-oxide-dft inspect runs/cu111_fcp/pw.in
grep -E "lfcp|assume_isolated|esm_bc|fcp_mu" runs/cu111_fcp/pw.in
```

You should see all four keys present in the namelist. The `fcp_mu` value should be close to `-(4.44 + (-0.4)) / 13.606 ≈ -0.297` Ry.

### 7b. (Optional) Smoke-test the SCF

Even if your stock QE supports ESM but doesn't fully converge FCP, the SCF step alone (without the FCP loop) should still parse the namelists without error:

```bash
cd runs/cu111_fcp && timeout 600 mpirun -n 4 pw.x -in pw.in > pw.out 2>&1 ; cd -
head -50 runs/cu111_fcp/pw.out   # Look for namelist parsing errors
```

If the output errors out at `lfcp` or `assume_isolated`, your QE build doesn't support those features and you need to wait for Frontier.

---

## Phase 8 — NEB reconstruction barriers (input-only locally)

NEB on a 16-atom slab × 5 images is borderline-feasible on a CPU box (hours, not days); for input verification, just check that `neb.x` parses our generated input.

### 8a. Generate a NEB input

```python
python -c "
from copper_oxide_dft.neb import write_neb_input
from copper_oxide_dft.structure_builder import build_cu111_slab, add_oxygen_adsorbates

slab = build_cu111_slab(supercell=(2, 2))
initial = add_oxygen_adsorbates(slab, coverage_ml=0.25, site='fcc')
final = add_oxygen_adsorbates(slab, coverage_ml=0.25, site='hcp')

write_neb_input(
    'runs/oads_diffusion/neb.in',
    endpoints=(initial, final),
    n_intermediate_images=3,
    pseudopotentials={'Cu': 'Cu.upf', 'O': 'O.upf'},
)
"
head -30 runs/oads_diffusion/neb.in
```

You should see `BEGIN_PATH_INPUT`, the `&PATH` namelist, then the engine block followed by `FIRST_IMAGE` / `LAST_IMAGE` blocks with `ATOMIC_POSITIONS angstrom`.

### 8b. (Optional) Run a short NEB

```bash
cd runs/oads_diffusion && mpirun -n 4 neb.x -in neb.in > neb.out 2>&1 ; cd -
tail -30 runs/oads_diffusion/neb.out
```

Don't expect convergence — five images × multi-iteration SCF will run for hours. Look for the line `path optimization step    1` to confirm `neb.x` accepted the input and started iterating. That's enough to validate the input format before scaling to Frontier.

---

## Final — Hand off to Frontier

Once the local validation is in hand:

```bash
copper-oxide-dft make-slurm runs/phase4 \
  --account <your-project> \
  --walltime 2:00:00 \
  --qe-module quantum-espresso/<version>-gpu
rsync -av runs/phase4/ frontier:scratch/phase4/
ssh frontier
for d in scratch/phase4/*/; do (cd "$d" && sbatch submit.sh); done
```

Once the Frontier outputs come back, the same `aggregate-pourbaix-energies` → `pourbaix --energies` flow you tested locally now produces the production diagram with real DFT+U numbers.

---

## A note on AMD RX 7900 XT GPUs

The 7900 XT/XTX is RDNA3 (`gfx1100`). ROCm 6.x has consumer-card support but **Quantum ESPRESSO's HIP port targets MI200/MI300 (`gfx90a` / `gfx940`)** — the architecture used on Frontier. Building Q-E for `gfx1100` is not officially supported and reports of success are anecdotal at best.

Recommendation: **use CPU mode locally**, treat the GPUs as available compute for ML / PyTorch / non-Q-E workloads, and run all GPU-accelerated DFT on Frontier where the hardware (`gfx90a`) is the tested target. Local CPU is fast enough for Phases 1–4 and Phase 8 smoke tests; only the explicit-water and full surface-coverage Phase 6 runs really need a cluster.

If you eventually want to try a HIP build for `gfx1100`:
- ROCm ≥ 6.0
- Q-E main branch (not 7.x release) — HIP support is most current there
- `./configure --enable-openmp --enable-hip GPU_ARCH=gfx1100`
- Expect compilation issues and pin them to specific Q-E commits when reporting

## Optional: building QE from source

If the apt version is too old, missing `hp.x`/`neb.x`, or you need the Environ patch:

```bash
sudo apt install -y libfftw3-dev libopenblas-dev libscalapack-mpi-dev
git clone https://gitlab.com/QEF/q-e.git
cd q-e
git checkout qe-7.3   # or whatever release
./configure --enable-openmp --enable-mpi
make -j$(nproc) pw neb hp
sudo make install   # or just add bin/ to PATH
```

For Environ, swap out the `make` step for the Environ-patched build (see [Phase 5](#phase-5--environ-implicit-solvation-input-only-locally)).

## Pitfalls

- **`pseudo_dir` is wrong** — the most common error. Make sure `$CUOXDFT_PSEUDO_DIR` is exported in the same shell you run the CLI from, and that the UPF filenames in the inputs (default `Cu.upf` / `O.upf` / `H.upf`) actually exist in that directory.
- **SCF not converging on CuO** — almost always means the AFM starting moments didn't survive into the input. Run `inspect` and confirm you see per-atom `starting_magnetization` lines for the Cu species; if not, the input writer was bypassed.
- **Wrong lattice parameter from vc-relax** — cutoff issue. Re-run Phase 1c with higher `ecutwfc`.
- **Phase 4 Pourbaix puts CuO stable everywhere** — either the U value is too low (oxide over-stabilized) or `mol_h2o` ran with a different cutoff/kpts than the bulk phases (DFT energies don't cancel cleanly). Re-check that all 5 systems used the same `ecutwfc`.
- **`pw.x: command not found`** — `sudo apt install quantum-espresso` or build from source. If you built from source, also add `q-e/bin/` to `PATH`.
- **Environ namelist errors at runtime** — your `pw.x` is not the Environ-patched build. `pw.x --version` won't tell you; you have to know which binary you compiled.
- **`lfcp` not recognized** — Phase 7 features depend on QE version. Try QE 7.2 or newer.
