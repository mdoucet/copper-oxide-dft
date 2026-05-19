# Startup Guide: CuO on Cu(111) in Non-Aqueous Electrolyte at −0.8 V

End-to-end workflow for a specific scientific question:

> *Why does a copper-oxide overlayer **remain** on a Cu(111) substrate
> under cathodic polarization (U = −0.8 V vs Ag/AgCl) in a non-aqueous
> electrolyte (THF + 1 % EtOH)?*

The plan: prototype Phases 1–2 (bulk Cu, bulk CuO) and the new MLIP-GCGO
structural-discovery pipeline on a **DGX Spark (NVIDIA GB10)**
workstation, then move the predicted top-K candidates to **ORNL
Frontier** for ESM-FCP reranks at constant U.

---

## TL;DR — the path

```text
Phase 1  bulk Cu convergence ................. DGX Spark, hours
Phase 2  bulk CuO + Hubbard U ................ DGX Spark, hours-overnight
Block C  DFT box-sampling dataset (~5k structs) DGX Spark, days
Block D  MACE-MP-0 fine-tune + validate ...... DGX Spark, overnight
Block E  GCGA unbiased + biased μ_O sweeps ... DGX Spark, days
Block F  Top-K ESM-FCP rerank at U = -0.8 V .. Frontier, days
Block G  SLD vs neutron-reflectometry ........ DGX Spark, hours
```

The methodology is locked in [ml-gcgo-pivot.md](ml-gcgo-pivot.md). The
implementation lives in [`src/copper_oxide_dft/ml/`](../src/copper_oxide_dft/ml/).
The DGX Spark install path is in [dgx-spark-ml-install.md](dgx-spark-ml-install.md).
The pre-pivot O-adsorbate ladder lives at the bottom of this document
([§10](#10-optional-sanity-check--cu111--o-adsorbate-ladder)) as an
*optional* sanity check, not the production answer.

---

## 0. Three deviations from the baseline implementation plan

The repo's [implementation-plan.md](implementation-plan.md) and
[local-workstation.md](local-workstation.md) assume **aqueous**
electrochemistry, a hand-built O-on-Cu(111) coverage ladder, and
**Frontier (AMD)** as the production target. Your workflow differs on
three axes. Read this section before any calculation.

### 0.1 Non-aqueous electrolyte ⇒ skip the aqueous Pourbaix, change the reference

The CHE machinery ([che.py](../src/copper_oxide_dft/che.py),
[pourbaix.py](../src/copper_oxide_dft/pourbaix.py)) is hard-coded to
H₂O as the proton reservoir and uses pH as an axis. Neither concept
applies in **THF with 1 % EtOH as proton donor**. You skip Phase 4
entirely and go from Phase 2 bulk-CuO directly into the MLIP-GCGO
pipeline (Blocks C–G).

The real-system numbers, all committed to
[ground_truths.md](ground_truths.md) (2026-05-18 entry):

| Quantity | Value | Notes |
|---|---|---|
| Solvent | **THF** | ε = 7.52 (CRC, 298 K). ~10× weaker screening than water; implicit-solvation shifts will be modest. |
| Proton donor | **1 % EtOH** | μ(H⁺ + e⁻) reference is EtOH ⇌ EtO⁻ + H⁺. Doesn't affect ESM-FCP at fixed U; matters only for PCET extensions. |
| Reference electrode | **Ag/AgCl** | Absolute = **4.64 V vs vacuum**. **Caveat:** pseudo-reference in non-aqueous; rigorous fix is calibration against Fc/Fc⁺. |
| Target potential | **U = −0.8 V vs Ag/AgCl** | μ_e = −(4.64 − 0.8) = **−3.84 eV vs vacuum**. |

Every ESM-FCP call passes `she_absolute_v=4.64`:

```python
from copper_oxide_dft.qe_input import fcp_overrides_for_potential
fcp_overrides_for_potential(-0.8, she_absolute_v=4.64)
# fermi_level_ev_vs_vacuum = -(4.64 - 0.8) = -3.84 eV
# fcp_mu (Ry)              = -3.84 / 13.6057 = -0.282 Ry
```

If you ever quote U vs aqueous SHE in a paper, **subtract 0.197 V**
from your Ag/AgCl readings.

### 0.2 "CuO on Cu" ⇒ MLIP-GCGO structural search, not a hand-built ladder

The earlier draft of this guide modelled "CuO on Cu" with a Cu(111)
slab carrying ¼ / ½ / ¾ / 1 ML O adsorbates. That hand-built ladder is
a *proxy*: Cu(111) + 1 ML O has 3-fold adsorbate coordination, real
CuO has 5-fold coordination. A 4-coverage scan cannot find a
reconstruction it wasn't seeded with.

The pivot ([ml-gcgo-pivot.md](ml-gcgo-pivot.md)) replaces the ladder
with a **machine-learned interatomic potential (MACE) + grand-canonical
genetic algorithm (GCGA)** workflow:

1. Box-sample perturbed Cu/Cu-O bulk seeds (Block C).
2. Run those through QE on DGX Spark to build a ~5 k-structure
   PBE-flavoured training set.
3. Fine-tune MACE-MP-0 medium on that set (Block D).
4. Drive a GCGA over μ_O ∈ [−7.0, −6.0] eV on a 12-layer Cu(111)
   substrate, lateral (4×4), with the fine-tuned MACE as the
   energy/force evaluator (Block E).
5. Reduce the resulting ~10 k-phase ensemble to a per-x_O minimum-Ω
   curve (Block E).
6. Take the top-K (K ≈ 20) candidates to Frontier, re-relax them under
   ESM-FCP at U = −0.8 V, and rerank by the constant-U grand potential
   ``Ω(U) = E_DFT + μ_e · tot_charge`` (Block F).
7. Convert each ensemble member to a Scattering Length Density (SLD)
   profile for comparison with the experimental neutron-reflectometry
   data (Block G).

This is more involved than the proxy ladder but it actually answers
"what wins" rather than "rank these four candidates."

### 0.3 DGX Spark (GB10) ⇒ no SLURM, CUDA-built QE, ARM userland; Frontier is rerank-only

GB10 is NVIDIA Grace+Blackwell (ARM CPU + Blackwell GPU, 128 GB unified
memory), a single workstation rather than a cluster. The repo's
`make-slurm` command (used in the original Phase-3 ladder workflow)
still targets Frontier for the ESM-FCP rerank; on DGX Spark you run
`pw.x` directly via the `qe-run` wrapper.

Two installs are needed:

- **Quantum ESPRESSO with CUDA** (Section 1.3 below): for the box-sampling
  DFT dataset and the ESM-FCP smoke test.
- **Python ML stack on top of the existing package** (Section 1.4 below;
  full details in [dgx-spark-ml-install.md](dgx-spark-ml-install.md)):
  torch + MACE + dscribe + scikit-learn + UMAP + h5py + GOCIA.

Once both work, the GCGA pipeline is `python -c "..."` on the
workstation.

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
sudo apt install -y nvhpc-25-x   # or current major
# Add to ~/.bashrc:
export NVHPC_ROOT=/opt/nvidia/hpc_sdk/Linux_aarch64/<version>
export PATH=$NVHPC_ROOT/compilers/bin:$NVHPC_ROOT/comm_libs/mpi/bin:$PATH
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
git checkout qe-7.3   # or current GPU-recommended release

./configure \
  --enable-openmp \
  --with-cuda=$NVHPC_ROOT/cuda \
  --with-cuda-cc=120 \
  --with-cuda-runtime=12.6   # match nvcc --version
make -j$(nproc) pw

# Optional but recommended:
make -j$(nproc) hp neb

# PATH:
export PATH=$(pwd)/bin:$PATH
which pw.x
pw.x --version | head -3
```

Verify GPU acceleration is actually wired in:

```bash
copper-oxide-dft bulk-cu --out /tmp/smoke/pw.in --ecutwfc 60
(cd /tmp/smoke && OMP_NUM_THREADS=4 mpirun -n 1 pw.x -in pw.in > pw.out)
grep -i "gpu" /tmp/smoke/pw.out | head
# Expect: "GPU acceleration is enabled" or similar.
```

If you see GPU lines, you're done. If not, re-configure with
`--with-cuda=...` explicit.

### 1.4 This package + the ML extras

```bash
git clone <your-fork>/copper-oxide-dft.git
cd copper-oxide-dft
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
pytest -q                 # expect 358 passed, 1 skipped (dscribe/umap path)

# Heavy ML stack — adds torch, MACE, dscribe, sklearn, UMAP, h5py.
pip install -e ".[ml]"
pip install git+https://github.com/zhouluo/GOCIA.git
```

Full bring-up checklist with smoke tests is in
[dgx-spark-ml-install.md](dgx-spark-ml-install.md).

### 1.5 Pseudopotentials

You need **Cu, O, and H** PseudoDojo PBE PAW pseudopotentials.

```bash
mkdir -p ~/pseudos
# Download Cu.upf, O.upf, H.upf from http://www.pseudo-dojo.org/
mv ~/Downloads/{Cu,O,H}.upf ~/pseudos/
echo 'export CUOXDFT_PSEUDO_DIR=~/pseudos' >> ~/.bashrc
```

### 1.6 The `qe-run` wrapper

```bash
mkdir -p ~/bin
cat > ~/bin/qe-run <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
work_dir="${1:?usage: qe-run <dir>}"
ranks="${QE_NRANKS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
cd "$work_dir"
if [[ "$ranks" -gt 1 ]] && command -v mpirun >/dev/null 2>&1; then
    mpirun -n "$ranks" pw.x -in pw.in > pw.out
elif command -v mpirun >/dev/null 2>&1; then
    mpirun -n 1 pw.x -in pw.in > pw.out
else
    pw.x -in pw.in > pw.out
fi
EOF
chmod +x ~/bin/qe-run
case ":$PATH:" in *":$HOME/bin:"*) ;; *) export PATH="$HOME/bin:$PATH" ;; esac
```

### 1.7 MACE foundation weights

```bash
mkdir -p ~/models
cd ~/models
curl -L -o 2023-12-03-mace-mp-0-medium.model \
  https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-03-mace-mp-0-medium.model
echo "export MACE_MP_0_MEDIUM=$HOME/models/2023-12-03-mace-mp-0-medium.model" >> ~/.bashrc
```

Smoke-test:

```bash
python - <<'PY'
import torch
from ase.build import molecule
from mace.calculators import mace_mp

print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
atoms = molecule("CO2")
atoms.calc = mace_mp(model="medium", device="cuda")
print(f"E(CO2) = {atoms.get_potential_energy():.4f} eV  (foundation-only)")
PY
```

`torch.cuda.is_available()` must print `True`. If `False`, your torch
wheel is CPU-only — see [dgx-spark-ml-install.md §1](dgx-spark-ml-install.md).

---

## 2. Phase 1 — Bulk Cu convergence

Identical to the [Ubuntu walkthrough §Phase 1](local-workstation.md#phase-1--bulk-cu).
You need it because every subsequent step (Phase 2, Block C dataset
generation) reads `configs/converged.json:bulk_cu` for `ecutwfc`,
`kpts`, `degauss`, and `lattice_a_ang`.

### 2.1 Single SCF sanity check

```bash
copper-oxide-dft bulk-cu --out runs/bulk_cu/pw.in
copper-oxide-dft inspect runs/bulk_cu/pw.in
qe-run runs/bulk_cu
copper-oxide-dft parse runs/bulk_cu/pw.out
```

`parse` should print `done=True` and a total energy. On GB10 with GPU
acceleration this finishes in seconds.

### 2.2 Convergence sweeps

```bash
# ecutwfc
copper-oxide-dft sweep --param ecutwfc --values 40,60,80,100,120,140 \
                       --out runs/conv_ecutwfc
for d in runs/conv_ecutwfc/*/; do qe-run "$d"; done
copper-oxide-dft sweep-analyze runs/conv_ecutwfc \
                               --threshold-mev 1 --png runs/conv_ecutwfc/convergence.png

# kpts
copper-oxide-dft sweep --param kpts --values 10,12,14,16,18,20,24 \
                       --out runs/conv_kpts
for d in runs/conv_kpts/*/; do qe-run "$d"; done
copper-oxide-dft sweep-analyze runs/conv_kpts \
                               --threshold-mev 1 --png runs/conv_kpts/convergence-kpts.png

# degauss
copper-oxide-dft sweep --param degauss --values 0.005,0.01,0.02,0.03,0.04,0.05,0.06,0.07 \
                       --out runs/conv_degauss
for d in runs/conv_degauss/*/; do qe-run "$d"; done
copper-oxide-dft sweep-analyze runs/conv_degauss \
                               --threshold-mev 1 --png runs/conv_degauss/convergence-degauss.png
```

### 2.3 Lock the converged triplet in `configs/converged.json`

```python
python -c "
from copper_oxide_dft.config import ProjectConfig, SystemConfig, save_config
cfg = ProjectConfig(systems={
    'bulk_cu': SystemConfig(
        ecutwfc_ry=<your-converged-ecutwfc>,
        kpts=(<n>, <n>, <n>),
        degauss_ry=<your-converged-degauss>,
        extras={'convergence_source': 'phase1-sweep <date>'},
    ),
})
save_config(cfg, 'configs/converged.json')
"
```

Phase 1 values committed for this project are documented in
[ground_truths.md](ground_truths.md) — search for *"Phase 1 converged
parameters"*.

### 2.4 Lattice parameter (vc-relax)

```bash
copper-oxide-dft bulk-cu --out runs/bulk_cu_vc/pw.in \
                          --calculation vc-relax --ecutwfc <converged>
qe-run runs/bulk_cu_vc
grep -A 4 "CELL_PARAMETERS" runs/bulk_cu_vc/pw.out | tail -4
```

PBE on Cu overshoots experimental `a = 3.615 Å` by ~1.2 % (relaxed
value lands around 3.658 Å). This is **inside the normal PBE range**;
the plan's "<0.5 %" criterion is unachievable with pure PBE and should
be read as ">3 % indicates a calculation bug." See
[ground_truths.md](ground_truths.md): *"PBE relaxed lattice parameter
for Cu"*.

Lock the relaxed value:

```python
python -c "
from copper_oxide_dft.config import load_config, save_config
cfg = load_config('configs/converged.json')
cfg.systems['bulk_cu'].extras['lattice_a_ang'] = <your-relaxed-a>
cfg.systems['bulk_cu'].extras['lattice_a_source'] = 'PBE vc-relax <date>'
save_config(cfg, 'configs/converged.json')
"
```

**Every Block ≥ C step that builds Cu / Cu(111) / Cu-O structures must
read this value, not the experimental 3.615 Å.**

---

## 3. Phase 2 — Bulk CuO + Hubbard U

Needed for two reasons:

1. **Hubbard U** value to use in DFT+U for Cu 3d — set in
   `configs/converged.json` and read everywhere downstream.
2. **Bulk-CuO seed structure** for the Block C box-sampling step. The
   GCGA candidates at high x_O should resemble CuO geometrically, and
   seeding the dataset with CuO ensures MACE sees the right
   coordination environment.

### 3.1 Generate the oxide bulks

```bash
copper-oxide-dft make-pourbaix-inputs runs/oxides \
                                       --ecutwfc <converged-from-phase-1> \
                                       --hubbard-u 4.0
# Writes bulk_cu, bulk_cu2o, bulk_cuo, mol_h2, mol_h2o.
copper-oxide-dft inspect runs/oxides/bulk_cuo/pw.in
```

Look for `starting_magnetization(1)` and `starting_magnetization(2)` on
the two Cu sub-species — AFM CuO needs both, and the writer has burned
us once before (see [ground_truths.md](ground_truths.md) 2026-05-14:
*AFM CuO species splitting*).

### 3.2 Run the AFM CuO SCF

```bash
qe-run runs/oxides/bulk_cuo
copper-oxide-dft parse runs/oxides/bulk_cuo/pw.out
```

Sanity:

- `job_done=True`
- `total_magnetization_bohr` near 0 (AFM cancels globally)
- Band gap in the output around 1.2–1.7 eV (experimental range)

If the gap is < 0.5 eV or total magnetization is large, the AFM
ordering didn't survive — re-check `starting_magnetization` and try
alternate moment patterns.

### 3.3 Hubbard U sweep (optional)

If you want a defensible U value rather than citing the literature:

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
```

Compare band gap to 1.2–1.7 eV experimental range; lock in your U in
[ground_truths.md](ground_truths.md) and `configs/converged.json` (use
`extras['hubbard_u_ev']`). The project default is **4.0 eV**.

---

## 4. Block C — DFT box-sampling ground-truth dataset

The MACE fine-tune (Block D) needs a diverse set of PBE-flavoured
labels covering the Cu/Cu-O composition space the GCGA will sweep over.
This is the most wall-clock-heavy step of the workflow (~5 k QE
relaxations × tens of minutes each ≈ days on a single GB10).

### 4.1 Box-sample perturbed structures

```python
python - <<'PY'
import numpy as np
from copper_oxide_dft.ml import BoxSamplingConfig, sample_batch
from copper_oxide_dft.structure_builder import (
    build_bulk_cu, build_bulk_cu2o, build_bulk_cuo,
)

# Seed bulks. Scale Cu to a small supercell so the post-perturbation cell
# has room to insert/delete oxygens without becoming pathological.
seeds = {
    "Cu":   build_bulk_cu() * (3, 3, 3),       # 27 Cu atoms
    "Cu2O": build_bulk_cu2o() * (2, 2, 2),     # 32 Cu + 16 O
    "CuO":  build_bulk_cuo() * (2, 2, 2),      # 32 Cu + 32 O
}

cfg = BoxSamplingConfig(
    rattle_stdev_ang=0.2,
    lattice_scale=0.05,
    max_o_insertions=8,
    max_o_deletions=8,
)
rng = np.random.default_rng(0)

all_results = []
seed_labels = []
for label, seed in seeds.items():
    # Aim for ~1500 accepted samples per seed → ~4500 total.
    batch = sample_batch(seed, n_samples=1500, config=cfg, rng=rng,
                          max_attempts_per_sample=10)
    all_results.extend(batch)
    seed_labels.extend([label] * len(batch))
    accepted = sum(1 for r in batch if r.accepted)
    print(f"{label:>5}: {accepted}/{len(batch)} accepted")

valid = [(r.atoms, label, r.info) for r, label in zip(all_results, seed_labels)
         if r.accepted]
print(f"Total accepted: {len(valid)}")
PY
```

Expected acceptance rate: 70–90 %. Lower than that means the Cu-O
connectivity filter is biting too aggressively — either widen
`cu_o_connectivity_cutoff_ang` or use larger seed supercells so
inserted O atoms have more space.

### 4.2 Write the QE inputs

```python
python - <<'PY'
import numpy as np
from copper_oxide_dft.config import load_config
from copper_oxide_dft.ml import (
    BoxSamplingConfig, sample_batch, write_dataset_inputs,
)
from copper_oxide_dft.structure_builder import (
    build_bulk_cu, build_bulk_cu2o, build_bulk_cuo,
)

cu_cfg = load_config("configs/converged.json").systems["bulk_cu"]
seeds = {
    "Cu":   build_bulk_cu() * (3, 3, 3),
    "Cu2O": build_bulk_cu2o() * (2, 2, 2),
    "CuO":  build_bulk_cuo() * (2, 2, 2),
}

cfg = BoxSamplingConfig()
rng = np.random.default_rng(0)

structs, labels, infos = [], [], []
for label, seed in seeds.items():
    for r in sample_batch(seed, 1500, cfg, rng):
        if r.accepted:
            structs.append(r.atoms)
            labels.append(label)
            infos.append(r.info)

entries = write_dataset_inputs(
    structs,
    out_root="runs/ml_dataset",
    system_config=cu_cfg,
    seed_labels=labels,
    perturbation_infos=infos,
    calculation="vc-relax",   # cell + atom positions; bulk box-sampling
)
print(f"Wrote {len(entries)} pw.in files under runs/ml_dataset/")
print(f"Manifest: runs/ml_dataset/manifest.jsonl")
print(f"Runner:   runs/ml_dataset/run_all.sh")
PY
```

What the writer encodes:

- Phase 1 converged `ecutwfc`, `degauss`.
- Γ-only k-points (perturbed 100+ atom supercells make a finite grid
  unaffordable; see [ml-gcgo-pivot.md §3.5](ml-gcgo-pivot.md)).
- `nosym=True`, `noinv=True` (perturbations break the seed cells' space
  groups; mandatory for any relaxation).
- Manuscript tolerances: `forc_conv_thr=1e-3 Ry/Bohr`,
  `conv_thr=1e-6 Ry`, `mixing_beta=0.3`.
- Spin polarisation + Hubbard U on Cu when O is present in the cell.

### 4.3 Run the batch

```bash
bash runs/ml_dataset/run_all.sh
```

The auto-generated `run_all.sh` is **resume-safe**: it skips any sample
directory whose `pw.out` already contains `JOB DONE`. Re-running it
after a crash or restart is free.

A 5 000-structure dataset at ~10 min/structure is **~35 days** of
wall-clock on a single GB10. Two strategies to keep the elapsed time
reasonable:

- Run fewer samples per seed (e.g. 500 each → ~12 days) and trust the
  UMAP subsampling in Block C.4 to maintain diversity.
- Split the seeds across multiple DGX Sparks if you have access to
  more than one; concatenate the manifests afterwards.

Don't try to run multiple `pw.x` instances concurrently on one
Blackwell GPU — `pw.x` saturates the GPU during the SCF cycle's dense
linear algebra; concurrent jobs serialise on the device and you lose
to context-switch overhead.

### 4.4 Curate the dataset

After the batch finishes, read the outputs back and run the manuscript
curation pipeline: force-filter → SOAP → IPCA → UMAP → 20×20 grid
subsample → 10:1 train/test split → extxyz.

```python
python - <<'PY'
from copper_oxide_dft.ml import (
    prepare_dataset, read_dataset_outputs,
)

items = read_dataset_outputs("runs/ml_dataset", require_job_done=True)
print(f"Read {len(items)} converged structures.")

split = prepare_dataset(
    items,
    train_path="runs/ml_dataset/cuox_train.extxyz",
    test_path="runs/ml_dataset/cuox_test.extxyz",
    max_force_ev_per_angstrom=10.0,
    grid_size=20,
    train_ratio=10.0 / 11.0,
    rng_seed=0,
)
print(split.summary())
PY
```

You want:

- `n_after_force_filter / n_input` ≥ 0.85 (15 % rejection rate is
  typical; >30 % means perturbations were too aggressive).
- `n_after_subsample` between 1 000 and 3 000 (smaller → not enough
  diversity for fine-tuning; much larger → wasted training time).
- `len(train) / len(test)` ≈ 10.

The extxyz files are ready to feed into MACE.

---

## 5. Block D — Fine-tune MACE-MP-0

### 5.1 Run the fine-tune

```bash
scripts/finetune_mace.sh \
    runs/ml_dataset/cuox_train.extxyz \
    runs/ml_dataset/cuox_test.extxyz \
    cuox_pbe_finetuned
```

The script wraps `mace_run_train` with the manuscript hyperparameters:
50 epochs, batch 4, lr 0.01, AMSGrad, EMA decay 0.99, float32, E0s
average, energy/forces weighted 1.0 each. Watch the loss curves with
`tensorboard --logdir checkpoints/cuox_pbe_finetuned/` (MACE writes
TensorBoard scalars by default).

On Blackwell with a ~2 000-structure curated training set, this is an
overnight run. CPU-only would take a couple of days; check
`nvidia-smi` if the wall time looks wrong.

### 5.2 Validate the fine-tuned model

```python
python - <<'PY'
from copper_oxide_dft.ml import evaluate_model_on_extxyz

metrics = evaluate_model_on_extxyz(
    model_path="cuox_pbe_finetuned.model",
    test_extxyz_path="runs/ml_dataset/cuox_test.extxyz",
    device="cuda",
)
print(metrics.summary())
print(f"Passes project targets? {metrics.passes_targets()}")
PY
```

Project targets ([ml-gcgo-pivot.md §6](ml-gcgo-pivot.md)):

- Energy MAE **< 30 meV/atom** (manuscript reports 9.8 on PBEsol; we're
  on PBE so 10–20 is expected, 20–30 is acceptable, >30 means
  pipeline trouble).
- Force MAE **< 100 meV/Å** (manuscript reports 35.3 on PBEsol).

If the energy MAE refuses to drop below 30:

- Bump `max_num_epochs` to 100 (manuscript ran 50; PBE may need more).
- Halve the learning rate to `0.005`.
- Don't touch the optimizer (AMSGrad) or batch size without re-reading
  the manuscript's ablation tables.
- Sanity-check the training data: are there pathological frames the
  force filter missed? Plot the energy histogram per seed; large
  outliers point at unphysical samples.

Lock the chosen model path in your environment:

```bash
echo "export CUOXDFT_MACE_MODEL=$PWD/cuox_pbe_finetuned.model" >> ~/.bashrc
```

---

## 6. Block E — Grand-canonical genetic algorithm

The structural search itself: ~10 k candidate phases produced by
sweeping the oxygen chemical potential μ_O over a 12-layer Cu(111)
substrate with the top 6 layers active.

### 6.1 Build the GCGA substrate

```python
python - <<'PY'
from copper_oxide_dft.config import load_config
from copper_oxide_dft.ml import GCGAConfig, build_cu111_gcga_substrate

cu = load_config("configs/converged.json").systems["bulk_cu"]
slab, active = build_cu111_gcga_substrate(
    layers=12,
    lateral=(4, 4),
    active_top_layers=6,
    lattice_a_ang=cu.extras["lattice_a_ang"],   # PBE-relaxed a, NOT 3.615
    vacuum_ang=20.0,
)
print(f"Substrate: {len(slab)} atoms, {len(active)} active.")
PY
```

The substrate carries a `FixAtoms` constraint on the bottom 6 layers;
GCGA's mutation routines respect it (only `active` atoms can be moved
or deleted, and insertions land on top of the active region).

### 6.2 Pin the GOCIA API on first run

The wrapper `copper_oxide_dft.ml.gcga.run_gcga_sweep` deliberately
raises `NotImplementedError`. The GOCIA package has reshuffled its
public API across releases; pin it once against the version installed
in your venv.

```bash
python -c "import gocia; print(gocia.__version__ if hasattr(gocia, '__version__') else 'src'); help(gocia)" \
    | head -40
```

Check which of these import paths actually works and update
[gcga.py](../src/copper_oxide_dft/ml/gcga.py) `run_gcga_sweep` to call
the matching API:

- `from gocia.popGen import evolve` (manuscript-era)
- `from gocia.geneticAlgorithm import run_ga` (newer)
- something else — read the GOCIA README.

The MACE calculator factory you'll plug in:

```python
from mace.calculators import MACECalculator
mace_calc = MACECalculator(model_paths=[model_path], device="cuda")

def evaluate(atoms):
    atoms.calc = mace_calc
    e = atoms.get_potential_energy()
    return biased_grand_potential_ev(e, atoms, config)
```

The math is already in [gcga.py](../src/copper_oxide_dft/ml/gcga.py)
(`grand_potential_ev`, `gaussian_bias_ev`, `biased_grand_potential_ev`,
`compute_x_o`). The substrate and active-index logic are also done.
Only the GOCIA glue is missing.

### 6.3 Unbiased μ_O sweep

```python
python - <<'PY'
import os
from pathlib import Path
from copper_oxide_dft.config import load_config
from copper_oxide_dft.ml import GCGAConfig, build_cu111_gcga_substrate
from copper_oxide_dft.ml.gcga import (
    DEFAULT_MU_O_RANGE_EV, DEFAULT_MU_O_N_POINTS, run_gcga_sweep,
)
import numpy as np

cu = load_config("configs/converged.json").systems["bulk_cu"]
slab, active = build_cu111_gcga_substrate(
    layers=12, lateral=(4, 4), active_top_layers=6,
    lattice_a_ang=cu.extras["lattice_a_ang"],
)

model = os.environ["CUOXDFT_MACE_MODEL"]
mu_o_grid = np.linspace(*DEFAULT_MU_O_RANGE_EV, DEFAULT_MU_O_N_POINTS)

for mu_o in mu_o_grid:
    out = Path(f"runs/gcga/unbiased/mu_o_{mu_o:+.2f}".replace(".", "p"))
    cfg = GCGAConfig(
        substrate=slab, active_indices=active, mu_o_ev=float(mu_o),
        n_generations=50, population_size=50,
        bias_centers=(),   # unbiased
    )
    print(f"μ_O = {mu_o:+.2f} eV → {out}")
    run_gcga_sweep(cfg, mace_model_path=model, out_dir=out, device="cuda")
PY
```

Each μ_O point runs ~2 500 MACE energy/force evaluations on the GPU
(50 generations × 50 population). MACE-MP-0 medium on Blackwell does
~100 inference/s for 200-atom cells, so each μ_O point is ~30 minutes
of wall time → the full 11-point unbiased sweep is **~6 hours**.

### 6.4 Biased x_O sweep

The unbiased sweep finds the natural minimum-Ω structure at each μ_O,
but skips over metastable intermediate stoichiometries. The biased
pass forces dense sampling across x_O ∈ [0.32, 1.0]:

```python
python - <<'PY'
import os, numpy as np
from pathlib import Path
from copper_oxide_dft.config import load_config
from copper_oxide_dft.ml import GCGAConfig, build_cu111_gcga_substrate
from copper_oxide_dft.ml.gcga import (
    DEFAULT_BIASED_X_O_RANGE, DEFAULT_BIASED_AMPLITUDE_EV,
    DEFAULT_BIASED_SIGMA, run_gcga_sweep,
)

cu = load_config("configs/converged.json").systems["bulk_cu"]
slab, active = build_cu111_gcga_substrate(
    layers=12, lateral=(4, 4), active_top_layers=6,
    lattice_a_ang=cu.extras["lattice_a_ang"],
)
model = os.environ["CUOXDFT_MACE_MODEL"]

bias_centers = tuple(np.linspace(*DEFAULT_BIASED_X_O_RANGE, 11))   # 11 bumps
for mu_o in (-6.5,):   # one μ_O is enough for the biased fill-in
    out = Path(f"runs/gcga/biased/mu_o_{mu_o:+.2f}".replace(".", "p"))
    cfg = GCGAConfig(
        substrate=slab, active_indices=active, mu_o_ev=mu_o,
        n_generations=50, population_size=50,
        bias_centers=bias_centers,
        bias_amplitude_ev=DEFAULT_BIASED_AMPLITUDE_EV,
        bias_sigma=DEFAULT_BIASED_SIGMA,
    )
    run_gcga_sweep(cfg, mace_model_path=model, out_dir=out, device="cuda")
PY
```

### 6.5 Build the ensemble

```python
python - <<'PY'
from pathlib import Path
from copper_oxide_dft.ml import (
    Phase, merge_ensembles, per_x_o_minima,
    read_ensemble_extxyz, write_ensemble_extxyz, top_k_by_omega,
)

# Each run_gcga_sweep call should write its final population as
# `<out>/population.extxyz` (this is what your GOCIA pin in §6.2 should
# produce). Pool them all here.
runs = list(Path("runs/gcga").rglob("population.extxyz"))
print(f"Found {len(runs)} GCGA populations.")

ensemble = []
for p in runs:
    ensemble.extend(read_ensemble_extxyz(p))
print(f"Total candidates: {len(ensemble)}")

merged = merge_ensembles(ensemble)
print(f"After dedup: {len(merged)}")

minima = per_x_o_minima(merged, n_bins=20, x_o_range=(0.0, 1.0))
print(f"Per-x_O minima (one per bin, may have empty bins): {len(minima)}")
write_ensemble_extxyz(minima, "runs/gcga/per_x_o_minima.extxyz")

top = top_k_by_omega(minima, k=20)
write_ensemble_extxyz(top, "runs/gcga/top20.extxyz")
print(f"Top-20 by Ω written to runs/gcga/top20.extxyz")
PY
```

A first look at the result before going to Frontier:

```python
python - <<'PY'
import matplotlib.pyplot as plt
from copper_oxide_dft.ml import read_ensemble_extxyz

minima = read_ensemble_extxyz("runs/gcga/per_x_o_minima.extxyz")
xs = [p.x_o for p in minima]
omegas = [p.omega_o_ev for p in minima]
plt.figure(figsize=(6, 4))
plt.plot(xs, omegas, "o-")
plt.xlabel(r"$x_O$"); plt.ylabel(r"$\Omega_O$ (eV)")
plt.title("GCGA per-$x_O$ minimum-$\\Omega$ curve")
plt.grid(True); plt.tight_layout()
plt.savefig("runs/gcga/omega_vs_xo.png", dpi=150)
PY
```

Look for:

- A monotonic-ish curve with possible kinks where reconstruction
  energetics shift.
- Coverage of x_O = [0, 1] with no large gaps (gaps mean GCGA didn't
  sample that composition — extend the biased pass).
- Reasonable per-x_O energies (Ω at x_O ≈ 0.5 should be lower than
  Ω at x_O = 1.0 if the GCGA is finding real Cu₂O-like minima rather
  than CuO-like ones, which is what you'd expect from the bulk
  stability hierarchy).

---

## 7. Block F — Top-K ESM-FCP rerank on Frontier

This is the step that finally answers your scientific question. The
top-K candidates from §6.5 get re-relaxed at fixed U = −0.8 V vs
Ag/AgCl, and ranked by the constant-U grand potential.

### 7.1 Generate ESM-FCP inputs

```python
python - <<'PY'
from copper_oxide_dft.config import load_config
from copper_oxide_dft.ml import (
    prepare_fcp_inputs, read_ensemble_extxyz, write_frontier_submit_scripts,
)

cu = load_config("configs/converged.json").systems["bulk_cu"]
top = read_ensemble_extxyz("runs/gcga/top20.extxyz")

paths = prepare_fcp_inputs(
    candidates=top,
    out_root="runs/fcp_rerank_minus0p8V",
    system_config=cu,
    u_target_v=-0.8,
    reference_absolute_v=4.64,    # Ag/AgCl absolute
    hubbard_u_ev=4.0,
    calculation="relax",          # NOT vc-relax: ESM-FCP wants a fixed cell
)
print(f"Wrote {len(paths)} pw.in files.")
PY
```

Spot-check one input:

```bash
copper-oxide-dft inspect runs/fcp_rerank_minus0p8V/candidate_00/pw.in
grep -E "lfcp|assume_isolated|esm_bc|fcp_mu" runs/fcp_rerank_minus0p8V/candidate_00/pw.in
# Expect: lfcp=.true., assume_isolated='esm', esm_bc='bc2',
#         fcp_mu ≈ -0.282 Ry (= -3.84 eV / 13.6057)
```

### 7.2 Wrap with Frontier SLURM scripts

```python
python -c "
from copper_oxide_dft.ml import write_frontier_submit_scripts
write_frontier_submit_scripts(
    out_root='runs/fcp_rerank_minus0p8V',
    account='<YOUR_PROJECT_ID>',
    walltime='4:00:00',
    qe_module='quantum-espresso/<version>-gpu',
)
"
```

Each `candidate_NN/submit.sh` runs `srun --gpus-per-task=1
--gpu-bind=closest pw.x ...` on Frontier's 8-GCD-per-node config.

### 7.3 Ship + submit + retrieve

```bash
# Ship.
rsync -av runs/fcp_rerank_minus0p8V/ frontier:scratch/fcp_rerank/
rsync -av ~/pseudos/ frontier:scratch/pseudos/   # first time only

# Submit on Frontier.
ssh frontier
echo "export CUOXDFT_PSEUDO_DIR=/lustre/.../pseudos" >> ~/.bashrc
for d in /lustre/.../scratch/fcp_rerank/candidate_*/; do
    (cd "$d" && sbatch submit.sh)
done

# Wait — these are ~hours per candidate; ESM-FCP convergence is the
# slowest part of the whole pipeline. Monitor:
squeue -u $USER

# Pull back when done.
exit   # back to DGX Spark
rsync -av frontier:/lustre/.../scratch/fcp_rerank/ runs/fcp_rerank_minus0p8V/
```

### 7.4 Verify the `tot_charge` regex against a real Frontier output

The constant-U Ω depends on `tot_charge` parsed from `pw.out`. The
default regex in
[fcp_rerank.py](../src/copper_oxide_dft/ml/fcp_rerank.py) takes the
*last* standalone `tot_charge = ...` line in the output. **This
heuristic must be verified on a real Frontier ESM-FCP run before you
trust the ranking.**

```bash
# Look at where `tot_charge` appears in one finished pw.out:
grep -n "tot_charge" runs/fcp_rerank_minus0p8V/candidate_00/pw.out
```

You expect to see:

- One occurrence in the input namelist echo near the top of the file
  (input `tot_charge` is whatever QE was started with; usually 0.0).
- N occurrences in the FCP iteration block as the loop converges.
- A final value that matches the converged FCP charge.

The regex's "last occurrence" heuristic picks the last of those, which
should be the converged one. If your QE version prints the converged
value *before* a final per-iteration trace line, the heuristic picks
the wrong number — tighten the regex to anchor on the FCP block
specifically.

### 7.5 Rank the candidates

```python
python - <<'PY'
from copper_oxide_dft.ml import rank_fcp_results

results = rank_fcp_results(
    out_root="runs/fcp_rerank_minus0p8V",
    u_target_v=-0.8,
    reference_absolute_v=4.64,
)
print(f"{'candidate':<14}{'E (eV)':>14}{'tot_charge':>14}{'Ω(U) (eV)':>14}")
print("-" * 56)
for r in results:
    q = f"{r.tot_charge:+.3f}" if r.tot_charge is not None else "  N/A"
    o = f"{r.omega_u_ev:+.4f}" if r.omega_u_ev is not None else "    N/A"
    print(f"{r.candidate_id:<14}{r.energy_ev:>+14.4f}{q:>14}{o:>14}")
PY
```

The candidate at the top of the table is the predicted surface state
at U = −0.8 V vs Ag/AgCl.

### 7.6 What the answer means

Three possible outcomes:

1. **A CuO-like coverage (x_O ≈ 0.5) wins thermodynamically.** Then
   "remains" is a thermodynamic statement — CuO is the predicted
   ground state at U = −0.8 V in non-aqueous, and the experimental
   observation is explained.
2. **A Cu-rich coverage (x_O ≪ 0.5) wins thermodynamically.** Then
   the answer is *kinetic*, not thermodynamic — CuO is metastable, and
   what stabilises it on the experimental timescale is a barrier to
   reduction. Next step: run NEB ([neb.py](../src/copper_oxide_dft/neb.py))
   from the experimental CuO terminus to the predicted Cu-like ground
   state and look at the activation energy.
3. **An intermediate suboxide (x_O ≈ 0.2–0.4) wins.** This is the most
   interesting outcome — it means the GCGA found a phase the
   experimental Pourbaix doesn't predict, and the next step is to
   characterise it structurally (cation arrangement, layer thickness)
   and check if the NR data could plausibly be re-fit as that phase.

---

## 8. Block G — SLD comparison to neutron reflectometry

The final, falsifiable step: convert each top-K candidate to a
neutron-reflectometry Scattering Length Density profile and overlay
the experimental NR data.

### 8.1 SLD of the predicted winner

```python
python - <<'PY'
from copper_oxide_dft.ml import (
    bulk_cu_normalization_factor,
    compute_sld_profile,
    rank_fcp_results,
)

# Pull the FCP-converged geometries (rank_fcp_results carries each
# candidate's relaxed structure via the `atoms` field).
results = rank_fcp_results("runs/fcp_rerank_minus0p8V", u_target_v=-0.8)
winner = results[0]
print(f"Winner: {winner.candidate_id}  Ω(U) = {winner.omega_u_ev:+.4f} eV")

# Lattice-correction factor for our PBE-overshoot.
norm = bulk_cu_normalization_factor(simulated_a_ang=3.6577)

profile = compute_sld_profile(winner.atoms, bin_width_ang=1.0)
# Apply the manuscript's bulk-Cu normalisation.
profile_sld_e6 = profile.sld_e6_per_a2 * norm
PY
```

### 8.2 Overlay against the experimental NR data

```python
python - <<'PY'
import matplotlib.pyplot as plt
import numpy as np

# Load your experimental SLD profile (whatever format your NR fitter
# spits out; here we assume a two-column z[Å], SLD[10⁻⁶ Å⁻²] CSV).
exp = np.loadtxt("experimental_nr.csv", delimiter=",")
plt.plot(exp[:, 0], exp[:, 1], "k-", label="Experiment (NR)")

# Top-3 predictions for visual context.
from copper_oxide_dft.ml import (
    compute_sld_profile, bulk_cu_normalization_factor, rank_fcp_results,
)
results = rank_fcp_results("runs/fcp_rerank_minus0p8V", u_target_v=-0.8)
norm = bulk_cu_normalization_factor(simulated_a_ang=3.6577)
for r in results[:3]:
    if r.atoms is None: continue
    prof = compute_sld_profile(r.atoms, bin_width_ang=1.0)
    plt.plot(prof.z_centres_ang, prof.sld_e6_per_a2 * norm,
             label=f"{r.candidate_id}  Ω={r.omega_u_ev:+.3f} eV")
plt.xlabel("z (Å)"); plt.ylabel("SLD (10⁻⁶ Å⁻²)")
plt.legend(); plt.tight_layout()
plt.savefig("runs/results/sld_overlay.png", dpi=200)
PY
```

What "good" looks like:

- Bulk-Cu region of all three profiles overlaps the experimental Cu
  SLD baseline (~6.5 × 10⁻⁶ Å⁻²). If it doesn't, your normalisation
  factor is wrong — re-check `simulated_a_ang`.
- The oxide-layer region (high-z) shows the SLD dropping below the Cu
  baseline (oxygen has a lower scattering length than Cu) at a depth
  consistent with what the experiment sees.
- The winner's profile is a better match than the runners-up.

---

## 9. What "good" looks like end-to-end

By the time you've worked through this guide:

- ✅ Phase 1 converged `ecutwfc` / kpts / degauss / `a` for Cu in
  `configs/converged.json`.
- ✅ Phase 2 Hubbard U on Cu 3d locked in `configs/converged.json` and
  documented in [ground_truths.md](ground_truths.md).
- ✅ Box-sampling DFT dataset of ~3 000–5 000 PBE-relaxed structures
  in `runs/ml_dataset/`.
- ✅ Fine-tuned MACE model with energy MAE < 30 meV/atom and force MAE
  < 100 meV/Å on the 1/11 hold-out.
- ✅ GCGA ensemble covering x_O ∈ [0, 1] with per-x_O minimum-Ω
  candidates in `runs/gcga/per_x_o_minima.extxyz`.
- ✅ Top-20 ESM-FCP-converged candidates at U = −0.8 V in
  `runs/fcp_rerank_minus0p8V/`.
- ✅ A single predicted winner with `Ω(U)` reported alongside the
  runners-up.
- ✅ SLD profile of the winner overlaid on the experimental NR data
  in `runs/results/sld_overlay.png`.
- ✅ A defensible answer to "why does CuO remain at U = −0.8 V?":
  thermodynamic, kinetic, or a previously-unsuspected suboxide.

---

## 10. Optional sanity-check — Cu(111) + O-adsorbate ladder

The pre-pivot proxy still exists in-tree as a sanity check. Don't run
it as your production answer (see [§0.2](#02-cuo-on-cu--mlip-gcgo-structural-search-not-a-hand-built-ladder)
for why) — but it's a cheap way to confirm your Phase 1/2 setup
produces sensible numbers before committing days to the GCGA pipeline.

```python
python - <<'PY'
from copper_oxide_dft.config import load_config
from copper_oxide_dft.qe_input import (
    DEFAULT_PSEUDOPOTENTIALS, merge_namelist_overrides,
    spin_and_hubbard_overrides, write_pw_input,
)
from copper_oxide_dft.structure_builder import (
    add_oxygen_adsorbates, build_cu111_slab,
)

cu = load_config("configs/converged.json").systems["bulk_cu"]
for coverage in (1/4, 1/2, 3/4, 1.0):
    slab = build_cu111_slab(layers=4, supercell=(3, 3), vacuum_ang=20.0,
                             a=cu.extras["lattice_a_ang"])
    covered = add_oxygen_adsorbates(slab, coverage_ml=coverage, site="fcc")
    label = f"{int(coverage * 100):03d}"

    overrides = merge_namelist_overrides(
        spin_and_hubbard_overrides(covered, nspin=2, hubbard_u={"Cu": 4.0}),
        {"system": {"nosym": True, "noinv": True}},
    )
    write_pw_input(
        covered,
        out_path=f"runs/proxy_cu111_O_{label}ML/pw.in",
        pseudopotentials=DEFAULT_PSEUDOPOTENTIALS,
        calculation="relax",
        kpts=(6, 6, 1),
        ecutwfc=cu.ecutwfc_ry,
        degauss=cu.degauss_ry,
        extra_input_data=overrides,
    )
PY

for d in runs/proxy_cu111_O_*/; do qe-run "$d"; done
```

Expected adsorption energy for ¼ ML O on Cu(111): around −4.5 to −5 eV
vs ½O₂. If your number is way off in the proxy, suspect (in order):
wrong O₂ reference, wrong spin treatment, unconverged cutoffs. If the
proxy looks reasonable, your Phase 1/2 is on solid ground and the
GCGA pipeline will inherit those numbers correctly.

---

## 11. Pitfalls specific to this workflow

- **"U vs Ag/AgCl is U vs SHE"** — it isn't. Off by +0.197 V. In
  non-aqueous, Ag/AgCl is a pseudo-reference that drifts. **Lock
  `she_absolute_v=4.64` in every ESM-FCP call site** and *never* mix
  references mid-project. If you later calibrate against Fc/Fc⁺,
  re-derive and update [ground_truths.md](ground_truths.md) in one
  pass.
- **"PBE on Cu lands at a = 3.615 Å"** — it doesn't. PBE relaxes to
  ~3.658 Å, a ~1.2 % overshoot. Every structure builder must read
  `lattice_a_ang` from `configs/converged.json`, not the experimental
  value.
- **"GCGA in vacuum gives me the answer at U = −0.8 V for free"** — it
  doesn't. The GCGA ranks by Ω_O at fixed μ_O; the constant-U ranking
  needs the ESM-FCP rerank on Frontier (Block F). If you read the
  GCGA's lowest-Ω structure and report it as "the answer at −0.8 V,"
  you're confusing two different free energies.
- **"More GCGA generations → better answer"** — not for free. MACE
  has a finite-MAE error envelope; running 1 000 generations chases
  noise once you're inside the MAE band. Stop at 50–100; spend the
  saved time on a wider μ_O sweep.
- **"The MACE test MAE doesn't matter once the GCGA is running"** —
  it absolutely does. If MAE = 25 meV/atom and the cell is 200 atoms,
  that's a 5 eV error envelope on Ω. Anything inside that envelope is
  noise; you may need to re-rerank the top-50 (not top-20) and let
  ESM-FCP filter the noise.
- **"ESM-FCP converged on iteration 1"** — suspicious. FCP should need
  several outer iterations to align the Fermi level. A 1-iteration
  "convergence" usually means `fcp_thr` is too loose.
- **"`tot_charge` is the converged FCP charge"** — only if the regex
  picks the right line. **Verify against a real Frontier pw.out
  before trusting Ω(U) for ranking** (see §7.4).
- **"The SLD profile looks great so we're done"** — only if the
  bulk-Cu region overlaps the experimental baseline *after* applying
  `bulk_cu_normalization_factor`. Without the normalisation, every
  SLD is biased by ~3.5 % from the PBE-lattice overshoot.
- **"GB10 GPU is just like Frontier"** — Blackwell ≠ MI250X.
  Performance scaling, peak memory, MPI rank counts are all different.
  Don't extrapolate GB10 wall times to Frontier; benchmark a small
  Frontier run first.
- **Pseudopotential filename mismatch** — defaults are now
  [`DEFAULT_PSEUDOPOTENTIALS`](../src/copper_oxide_dft/qe_input.py)
  (Cu.upf, O.upf, H.upf). PseudoDojo downloads with longer names;
  rename or override the dict everywhere.

---

## Related docs

- [ml-gcgo-pivot.md](ml-gcgo-pivot.md) — methodology lock for the
  MLIP-GCGO pivot; read this *before* changing any default in the
  pipeline.
- [machine-learned-dft.md](machine-learned-dft.md) — reference
  manuscript walkthrough this workflow reproduces.
- [dgx-spark-ml-install.md](dgx-spark-ml-install.md) — DGX Spark
  install checklist for the ML extras.
- [implementation-plan.md](implementation-plan.md) — full 9-phase
  roadmap; Phase 3 is now superseded for the central question by
  Blocks C–G.
- [local-workstation.md](local-workstation.md) — Ubuntu CPU walkthrough
  with aqueous defaults; cross-reference for any phase this guide is
  brief on (Phases 1–2 in particular).
- [ground_truths.md](ground_truths.md) — methodology decisions, AFM CuO
  gotchas, Frontier conventions, MLIP-GCGO pivot summary.
- [project.md](project.md) — project scope, dependencies, MLIP-GCGO
  example usage.
