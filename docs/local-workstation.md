# Local Workstation Walkthrough (Ubuntu, CPU)

End-to-end Phase 1 walkthrough on an Ubuntu workstation. The target hardware is a Linux box with multi-core CPU; this guide intentionally **does not** use the Radeon RX 7900 XT GPUs for Quantum ESPRESSO (see [GPU notes](#a-note-on-amd-rx-7900-xt-gpus) at the end). All cluster-bound work happens later on Frontier.

The point of running locally is to **validate the toolchain and the structures** before any of it touches Frontier hours. Bulk Cu is small enough that even one CPU thread converges in seconds, so this is the fastest way to confirm:

1. The pipeline produces well-formed `pw.x` inputs.
2. Those inputs match a known reference (fcc Cu, lattice parameter ~3.61 A).
3. The convergence sweep and parsing work end to end.
4. Structures look correct layer by layer (matters more as soon as slabs land in Phase 3).

## 0. Prerequisites

- Ubuntu 22.04 LTS or newer.
- Python 3.10+.
- `git`, `build-essential`, `gfortran`, `mpich` or `openmpi`.

```bash
sudo apt update
sudo apt install -y build-essential gfortran python3.10-venv git
```

## 1. Install Quantum ESPRESSO (CPU)

The simplest option is the distribution package:

```bash
sudo apt install -y quantum-espresso
which pw.x   # should print /usr/bin/pw.x
pw.x --version | head -3
```

This is typically QE 6.7-7.0 (older but feature-complete for our Phase 1 needs). If you want a newer build, see [Building QE from source](#optional-building-qe-from-source) below.

Smoke-test the install with the trivial example:

```bash
echo "&CONTROL
calculation='scf'
/
&SYSTEM
ibrav=0
nat=1
ntyp=1
ecutwfc=20
/
&ELECTRONS
/
ATOMIC_SPECIES
Cu 63.546 Cu.upf
ATOMIC_POSITIONS angstrom
Cu 0 0 0
K_POINTS gamma
CELL_PARAMETERS angstrom
2 0 0
0 2 0
0 0 2" > /tmp/smoke.in
pw.x -in /tmp/smoke.in > /tmp/smoke.out 2>&1 || head -20 /tmp/smoke.out
```

You should see `JOB DONE.` at the end (or a "pseudopotential file not found" error, which is expected before the next step).

## 2. Install this package

```bash
git clone <your-fork>/copper-oxide-dft.git
cd copper-oxide-dft
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
pytest -q   # should print "48 passed"
```

## 3. Pseudopotentials

Grab the Cu pseudopotential from [PseudoDojo](http://www.pseudo-dojo.org/):

- Functional: **PBE**
- Type: **scalar-relativistic**
- Accuracy: **standard**
- Format: **PAW (UPF)**

Drop it next to the project and point an env var at it:

```bash
mkdir -p ~/pseudos
mv ~/Downloads/Cu.upf ~/pseudos/
export CUOXDFT_PSEUDO_DIR=~/pseudos
echo 'export CUOXDFT_PSEUDO_DIR=~/pseudos' >> ~/.bashrc
```

## 4. Generate a single bulk-Cu input and verify the structure

```bash
copper-oxide-dft bulk-cu --out runs/bulk_cu/pw.in
copper-oxide-dft inspect runs/bulk_cu/pw.in
```

You should see something like:

```
File:        runs/bulk_cu/pw.in
Composition: Cu (1 atoms)
Volume:      11.8104 A^3
Cell vectors (A):
  a: (   0.0000    1.8075    1.8075)  |a| = 2.5562
  b: (   1.8075    0.0000    1.8075)  |b| = 2.5562
  c: (   1.8075    1.8075    0.0000)  |c| = 2.5562

Layers grouped by z (tol=0.1 A):
  [ 0] z =   0.0000 A  thickness = 0.0000 A  Cux1  (1 atoms)
```

Sanity check: `|a|^3 * sqrt(2) ~= 3.615^3 ~= 47.2 A^3`. We see 11.81 A^3 for the primitive cell, which is `47.2 / 4` -- correct for the fcc primitive cell (1 atom per cell, 4 atoms per conventional cubic cell).

> The `inspect` view becomes much more valuable for slabs in Phase 3. For a 4-layer Cu(111) slab you would see four `z = ...` lines listing Cu counts per layer; for an oxide-covered slab the top layers would mix Cu and O. That is the cheap, visual structural verification this command is for.

## 5. Run the SCF locally

```bash
cd runs/bulk_cu
pw.x -in pw.in > pw.out
# Multi-threaded (much faster on multi-core boxes):
# mpirun -n 4 pw.x -in pw.in > pw.out
cd -
```

This takes seconds. Then:

```bash
copper-oxide-dft parse runs/bulk_cu/pw.out
```

Expected output ends with `done=True`. Note the total energy (in Ry); we will compare it across the convergence sweep next.

## 6. Phase 1 convergence sweep (local, small)

A real convergence sweep on Frontier covers `{40, 60, 80, 100}` Ry but we do not need that much locally for validation. A coarse sweep is enough:

```bash
copper-oxide-dft sweep \
  --param ecutwfc --values 40,60,80 \
  --out runs/conv_ecutwfc

# Run each point locally.
for d in runs/conv_ecutwfc/*/; do
    (cd "$d" && pw.x -in pw.in > pw.out)
done

# Parse all of them.
copper-oxide-dft parse runs/conv_ecutwfc/*/pw.out
```

You should see total energy converge towards a fixed value as `ecutwfc` increases. The difference between 80 and 100 Ry should be smaller than 1 meV/atom on Frontier; on this small local sweep we are mostly checking that the relationship is monotonic and that the parser sees `done=True` for every point.

## 7. Lattice-parameter validation (vc-relax)

To prove the methodology gives the right answer for a system we already know cold, run a variable-cell relaxation on bulk Cu and compare to experiment (3.615 A):

```bash
copper-oxide-dft bulk-cu --out runs/bulk_cu_vc/pw.in --calculation vc-relax --ecutwfc 80
cd runs/bulk_cu_vc && pw.x -in pw.in > pw.out && cd -
copper-oxide-dft parse runs/bulk_cu_vc/pw.out
```

The relaxed `CELL_PARAMETERS` block in `pw.out` should give a cubic-equivalent lattice parameter within ~0.5 % of 3.615 A. (Right now you can extract it by grepping `CELL_PARAMETERS` from `pw.out`; we will add a relaxed-geometry reader to `parse.py` in the next iteration.) If the relaxed `a` is significantly off, something is wrong with the pseudopotential or the cutoffs -- stop and debug before moving to bigger systems.

## 8. Hand off to Frontier

Once the local validation is in hand:

```bash
copper-oxide-dft make-slurm runs/conv_ecutwfc \
  --account <your-project> \
  --qe-module quantum-espresso/<version>-gpu
rsync -av runs/conv_ecutwfc/ frontier:scratch/conv_ecutwfc/
ssh frontier
for d in scratch/conv_ecutwfc/*/; do (cd "$d" && sbatch submit.sh); done
```

## A note on AMD RX 7900 XT GPUs

The 7900 XT/XTX is RDNA3 (`gfx1100`). ROCm 6.x has consumer-card support but **Quantum ESPRESSO's HIP port targets MI200/MI300 (`gfx90a` / `gfx940`)** -- the architecture used on Frontier. Building Q-E for `gfx1100` is not officially supported and reports of success are anecdotal at best.

Recommendation: **use CPU mode locally**, treat the GPUs as available compute for ML / PyTorch / non-Q-E workloads, and run all GPU-accelerated DFT on Frontier where the hardware (`gfx90a`) is the tested target. Local CPU is fast enough for Phase 1 bulk Cu and useful as a correctness check before Frontier hours are spent.

If you eventually want to try a HIP build for `gfx1100`:
- ROCm >= 6.0
- Q-E main branch (not 7.x release) -- HIP support is most current there
- `./configure --enable-openmp --enable-hip GPU_ARCH=gfx1100`
- Expect compilation issues and pin them to specific Q-E commits when reporting

## Optional: building QE from source

If the apt version is too old for a feature you need (rare in Phase 1):

```bash
sudo apt install -y libfftw3-dev libopenblas-dev libscalapack-mpi-dev
git clone https://gitlab.com/QEF/q-e.git
cd q-e
git checkout qe-7.3   # or whatever release
./configure --enable-openmp --enable-mpi
make -j$(nproc) pw
sudo make install   # or just add bin/ to PATH
```

This gives a CPU build with MPI + OpenMP that often outpaces the apt version on multi-socket boxes.

## Pitfalls

- **`pseudo_dir` is wrong** -- the most common Phase 0 error. Make sure `$CUOXDFT_PSEUDO_DIR` is exported in the same shell you run the CLI from, and that the UPF filename in the input (default `Cu.upf`) actually exists in that directory.
- **SCF not converging** -- on bulk Cu this should never happen. If it does, the pseudopotential is suspect (re-download from PseudoDojo) or the cell is degenerate (rare; check `inspect`).
- **Wrong lattice parameter from vc-relax** -- almost always a cutoff issue. Run an `ecutwfc` sweep first.
- **`pw.x: command not found`** -- you forgot `sudo apt install quantum-espresso` or did not source `venv/bin/activate` (the QE binary itself is system-wide, but if you built from source it lives in `q-e/bin/pw.x` and needs to be on `PATH`).
