# DGX Spark — MLIP-GCGO install

Bring-up checklist for the MLIP-GCGO pivot on the DGX Spark (NVIDIA
Grace + Blackwell, ARM64, 128 GB unified memory). Run these once;
verify each step works before moving on.

For the *scientific* context of why we are doing this, see
[ml-gcgo-pivot.md](ml-gcgo-pivot.md). For the (now-superseded) static
Phase 3 install, see [startup-cuo-cu-nonaqueous.md §1](startup-cuo-cu-nonaqueous.md).

## 0. Prerequisites already in place from the static workflow

These were set up in [startup-cuo-cu-nonaqueous.md](startup-cuo-cu-nonaqueous.md);
none change for the pivot:

- DGX OS base (`uname -m` → `aarch64`).
- NVHPC SDK on PATH (`nvcc --version` works).
- CUDA-aware Quantum ESPRESSO (`pw.x --version` works, GPU lines
  visible in a smoke-test SCF).
- PseudoDojo PBE PAW pseudopotentials in `$CUOXDFT_PSEUDO_DIR`.
- This package installed (`pip install -e ".[dev]"` clean; pytest
  passes).

If any of those is missing, fix it before continuing.

## 1. Install the ML extras

```bash
cd ~/git/copper-oxide-dft
source venv/bin/activate

# Heavy install — torch + mace + dscribe pull a lot of native libs.
pip install -e ".[ml]"

# GOCIA is not on PyPI; install from source.
pip install git+https://github.com/zhouluo/GOCIA.git

# Verify
python - <<'PY'
import torch, mace, dscribe, umap, h5py, sklearn, scipy
import gocia
print(f"torch   : {torch.__version__}  cuda={torch.cuda.is_available()}")
print(f"mace    : {mace.__version__}")
print(f"dscribe : {dscribe.__version__}")
print(f"umap    : {umap.__version__}")
print(f"sklearn : {sklearn.__version__}")
print(f"gocia   : {getattr(gocia, '__version__', 'src checkout')}")
PY
```

`torch.cuda.is_available()` must print `True`. If `False` on Blackwell,
the torch wheel was the CPU-only build — re-install via the matching
CUDA wheel:

```bash
pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch torchvision torchaudio
```

(Use the cu12.x wheel that matches the NVHPC CUDA runtime in
`nvcc --version`.)

## 2. Download MACE-MP-0 medium foundation weights

```bash
mkdir -p ~/models
cd ~/models
# Pinned snapshot; the MACE team versions these by date.
curl -L -o 2023-12-03-mace-mp-0-medium.model \
  https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-03-mace-mp-0-medium.model

# Smoke-test the foundation model on a single CO2 molecule (cheap).
python - <<'PY'
from ase.build import molecule
from mace.calculators import mace_mp

atoms = molecule("CO2")
atoms.calc = mace_mp(model="medium", device="cuda")
print(f"E(CO2) = {atoms.get_potential_energy():.4f} eV (foundation-only)")
PY
```

Expect ~`-25.4 eV` order of magnitude. The number itself isn't
load-bearing — running without error and a non-NaN energy is.

Record the path:

```bash
echo "export MACE_MP_0_MEDIUM=$HOME/models/2023-12-03-mace-mp-0-medium.model" >> ~/.bashrc
```

## 3. GOCIA smoke test

```bash
python - <<'PY'
from gocia.interface import Interface
from ase.build import bulk
import gocia.popGen.geneticOps as ops

cu_bulk = bulk("Cu", cubic=True)
print(f"GOCIA Interface API present: {hasattr(Interface, '__init__')}")
print(f"GOCIA ops module present:    {hasattr(ops, 'mut_rattle')}")
PY
```

The API surface evolves; if the import fails on `gocia.popGen.geneticOps`,
the package layout changed — check `python -c "import gocia; help(gocia)"`
and update `src/copper_oxide_dft/ml/gcga.py` to match. The pivot doc
assumes the manuscript-era API.

## 4. Verify the existing test suite still passes

The ML extras pull torch + sklearn + scipy; these can perturb the
non-ML test suite if a version mismatch happens. Confirm nothing
regressed:

```bash
pytest -q   # expect the existing pass count (was 165) plus any new ML tests
```

If pytest collection errors out on a new module, that's a `src/`
package-discovery issue, not an env issue — re-run with `-x` and read
the traceback.

## 5. Environment summary to keep handy

After everything above:

```bash
cat <<'EOF' > ~/.cuoxdft_ml_env
# Source this before any MLIP-GCGO command.
export NVHPC_ROOT=/opt/nvidia/hpc_sdk/Linux_aarch64/<version>
export PATH=$NVHPC_ROOT/compilers/bin:$NVHPC_ROOT/comm_libs/mpi/bin:$PATH
export LD_LIBRARY_PATH=$NVHPC_ROOT/compilers/lib:$LD_LIBRARY_PATH
export CUOXDFT_PSEUDO_DIR=~/pseudos
export MACE_MP_0_MEDIUM=$HOME/models/2023-12-03-mace-mp-0-medium.model
source ~/git/copper-oxide-dft/venv/bin/activate
EOF
echo 'source ~/.cuoxdft_ml_env' >> ~/.bashrc
```

## 6. What can still bite you

- **aarch64 wheel availability.** torch ≥ 2.2 ships aarch64+CUDA
  wheels, but `dscribe` and `umap-learn` may need a native compile.
  If `pip install -e ".[ml]"` hangs on a compile step, install
  `apt install -y build-essential libomp-dev` and retry.
- **MACE foundation-model URL drift.** ACEsuit re-releases periodically.
  If the curl in §2 404s, find the latest release at
  `https://github.com/ACEsuit/mace-mp/releases` and update the URL in
  this doc.
- **GOCIA installed but import names different.** The package has had a
  few naming reshuffles. If `from gocia.popGen ...` doesn't work, check
  `from gocia.geneticAlgorithm ...`. Whichever import path works gets
  pinned in `src/copper_oxide_dft/ml/gcga.py`.
- **`torch.cuda.is_available()` returns True but `mace_mp(device="cuda")`
  silently runs on CPU.** Watch the GPU with `nvidia-smi` during the
  smoke test. If GPU utilisation stays at 0, MACE fell back to CPU and
  the GCGA will be ~50× slower than expected; rebuild torch against the
  exact CUDA runtime your NVHPC SDK ships.
