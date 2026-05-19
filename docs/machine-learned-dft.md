Shifting this workflow from static DFT to an ML-driven grand canonical search maps perfectly onto a standalone DGX Spark workstation. Since we are moving heavily into AI/ML tooling and custom local deployments, you can leverage the Grace+Blackwell architecture entirely in-house for both the interatomic potential training and the massive genetic algorithm inferences, deferring Frontier only if you need to rapidly scale the initial DFT dataset generation.

Here is your implementation plan to independently reproduce the MLIP-GCGO methodology described by your collaborators.

### Phase 1: Environment and Orchestration Setup

The new approach completely replaces the manual slab-building in Phase 3 of your original guide. You will need to deploy two major computational frameworks on your DGX:

1. 
**GOCIA Package:** This is the orchestration engine for both the training data generation (box-sampling) and the Grand Canonical Genetic Algorithm (GCGA) searches.


2. **MACE Framework:** The machine learning interatomic potential architecture. You will be using the `mace_run_train` interface to fine-tune the pre-trained `MACE-MP-0` foundation model.



### Phase 2: Generating the Ground Truth (DFT Box-Sampling)

You need to generate a diverse structural dataset of non-stoichiometric copper oxides.

1. 
**Initialization:** Use GOCIA to generate supercells of Cu, Cu8O, Cu2O, Cu4O3, CuO, and c-CuO.


2. 
**Perturbation:** Apply random rattling ($0.2\text{ \AA}$) and isotropic lattice scaling ($\pm5\%$). Randomly insert/remove up to 8 oxygen atoms for bulk, and up to 2 Cu/2 O for surface slabs, enforcing Cu-O connectivity.


3. 
**Pre-optimization:** Use a Hookean potential to correct unphysical atomic overlaps before passing to the DFT solver.


4. **DFT Relaxation:** Run local optimizations on these perturbed structures.
* 
*Note for reproduction:* The collaborators used VASP 6.4.1 with the PBEsol functional and a $400\text{ eV}$ cutoff at the $\Gamma$-point only.





### Phase 3: Descriptor-Based Filtering and MLIP Fine-Tuning

A dataset of highly perturbed structures will contain physical redundancies and high-energy artifacts.

1. 
**Force Filtering:** Strip out any trajectory frames where the maximum atomic force exceeds $10\text{ eV/\AA}$.


2. 
**Subsampling:** Featurize the remaining structures using Smooth Overlap of Atomic Positions (SOAP) descriptors. Compress these via Incremental PCA (50 components) and project them into a 2D space using UMAP. Lay a 20x20 grid over this projection and sample evenly to ensure your final training set covers the entire configuration space without redundancy.


3. 
**Fine-Tuning:** Split the data (10:1 train/test). Fine-tune `MACE-MP-0` for 50 epochs using a batch size of 4, a learning rate of 0.01, and an AMSGrad optimizer.



### Phase 4: Grand Canonical Genetic Algorithm (GCGA)

With the MACE MLIP acting as your highly accurate, blazingly fast energy/force evaluator, you can now search for the true ground state.

1. 
**The System:** Build a 12-layer Cu(111) supercell (approximately $3\text{ nm}$ thick), designating the top 6 layers as the active region where oxygen will be manipulated.


2. 
**Unbiased Searches:** Run GOCIA's GCGA across a sweep of target oxygen chemical potentials ($\mu_O$) from $-6.0\text{ eV}$ to $-7.0\text{ eV}$. The algorithm will handle crossover, mutation (rattling, O insertion/deletion), and selection based on the Grand Potential $\Omega_O$.


3. **Biased Searches:** The unbiased search will skip over metastable, intermediate-coverage states. You must run biased searches by adding a Gaussian potential to the fitness function, forcing the algorithm to densely sample stoichiometries ($x_O$) between 0.32 and 1.00.


4. 
**Merge:** Combine all results into a final ensemble of roughly 26,000 phases.



### Phase 5: SLD Simulation and Free Energy Mapping

The final step is translating the atomic coordinates of your ensemble into the experimental Neutron Reflectometry observables.

1. 
**Extract the Minimum Path:** From your merged ensemble, identify the lowest-$\Omega_O$ surface structure for each stoichiometry $x_O$.


2. 
**Calculate SLD:** For each phase, isolate a $10\text{ \AA}$ thick slab representing the interfacial region. Compute the SLD using $SLD = \sum(b_i)/(A \delta z)$, where $b_i$ are the coherent neutron scattering lengths for Cu and O, $A$ is the lateral area, and $\delta z$ is $10\text{ \AA}$.


3. 
**Normalization:** Because DFT lattice constants deviate slightly from true experimental values, you must normalize your calculated SLDs by the ratio of the experimental metallic Cu SLD to your simulated bulk Cu SLD.




### The ASE-Espresso Generation Script

Save this as `ml_data_generator.py` alongside your `configs/converged.json` file.

```python
import os
from ase.calculators.espresso import Espresso
from copper_oxide_dft.config import load_config

# 1. Load your Phase 1 Ground Truths
config = load_config("configs/converged.json")
cu_config = config.systems["bulk_cu"]
ECUTWFC = cu_config.ecutwfc_ry
# Note: For massive 5x5x12 supercells (300+ atoms), Gamma-point (1,1,1) is physical. 
# If GOCIA generates smaller primitive cells, you should use cu_config.kpts instead.

PSEUDO_DIR = os.environ.get("CUOXDFT_PSEUDO_DIR", os.path.expanduser("~/pseudos"))

def relax_structure_with_qe(atoms, run_dir, label, is_bulk=False):
    """
    Takes a perturbed structure from GOCIA, attaches a QE calculator, 
    and relaxes it to the manuscript's tight tolerances using your PBE baseline.
    """
    os.makedirs(run_dir, exist_ok=True)
    
    # 2. Define the QE Calculator
    calc = Espresso(
        pseudopotentials={'Cu': 'Cu.upf', 'O': 'O.upf'},
        pseudo_dir=PSEUDO_DIR,
        tstress=True,
        tprnfor=True,
        kpts=(1, 1, 1), 
        input_data={
            'control': {
                'calculation': 'vc-relax' if is_bulk else 'relax',
                'forc_conv_thr': 1.0e-3, # ~2.5e-2 eV/A (manuscript tolerance)
                'outdir': run_dir,
                'prefix': label
            },
            'system': {
                'ecutwfc': ECUTWFC, # Uses your converged Phase 1 cutoff
                'nosym': True,      # Critical: Perturbed structures break space groups
                'noinv': True,
            },
            'electrons': {
                'conv_thr': 1.0e-6, # ~10^-5 eV (manuscript tolerance)
                'mixing_beta': 0.3, # Aggressive dampening for highly disordered starting guesses
                'electron_maxstep': 100
            }
        },
        # Map directly to your DGX Spark FCP runner configuration
        command=f"mpirun -n 1 pw.x -in PREFIX.pwi > PREFIX.pwo" 
    )
    
    atoms.calc = calc
    
    # 3. Execute and capture
    try:
        # get_potential_energy() triggers the QE execution via ASE
        energy = atoms.get_potential_energy() 
        forces = atoms.get_forces()
        print(f"[{label}] Converged. E = {energy:.3f} eV")
        return atoms
    except Exception as e:
        print(f"[{label}] Convergence failed: {e}")
        return None

```

### Execution Strategy on DGX Spark

Because you are running on a single, highly capable Grace+Blackwell node, you want to manage the orchestration carefully so you don't stall the GPU or run out of memory.

1. **Batch Generation:** Configure GOCIA to generate the perturbed structures in batches (e.g., 500 at a time).
2. **Sequential GPU Execution:** Loop the `relax_structure_with_qe` function over the batch. Because `pw.x` will fully saturate the Blackwell GPU during the dense linear algebra steps of the SCF cycle, do *not* try to run multiple ASE calculators concurrently via Python multiprocessing. Let `mpirun -n 1 pw.x` own the GPU sequentially.
3. **Data Extraction:** Once a batch is complete, you will need a quick parser to strip out the failed convergences and the frames that violate the force threshold.

```bash
# A quick post-processing check to flag high-force frames before MACE training
grep -E "Total force" runs/batch_01/*/pw.out | awk '{if ($4 > 0.38) print $0}' 
# 0.38 Ry/Bohr is roughly the 10 eV/Å cutoff mentioned in the manuscript.

```

By generating this dataset with your specific `ecutwfc` and PBE pseudopotentials, the MACE model you fine-tune in the next step will inherently understand the exact energetic landscape of your baseline FCP parameters.



Let's map out the fine-tuning pipeline. To strictly reproduce the methodology from the supplementary information, we need to convert your Quantum ESPRESSO outputs into the Extended XYZ (`.extxyz`) format that MACE expects, apply the specific filtering criteria, and map the authors' hyperparameters directly to the `mace_run_train` command.

Here is the step-by-step implementation for your Grace+Blackwell node.

### 1. Data Preparation (The `.extxyz` Bridge)

Before feeding the data to MACE, you must filter the structures and convert them. The manuscript specifies removing frames where the maximum atomic force exceeds 10 eV/Å , and splitting the curated dataset into training and test sets using a 10:1 ratio.

Here is a quick ASE script to process the outputs from your QE batch generation:

```python
import numpy as np
from ase.io import read, write
import glob

# 1. Load all converged structures from your QE runs
pwo_files = glob.glob("runs/batch_*/*/pw.out")
valid_atoms = []

for out in pwo_files:
    try:
        # read() extracts the final geometry, energy, and forces
        atoms = read(out, format="espresso-out")
        
        # [cite_start]2. Apply the Force Filter (Max force < 10 eV/A) [cite: 118, 190]
        max_force = np.max(np.linalg.norm(atoms.get_forces(), axis=1))
        if max_force < 10.0:
            valid_atoms.append(atoms)
    except Exception:
        pass # Skip unconverged runs

print(f"Extracted {len(valid_atoms)} valid structures.")

# [cite_start]3. Train/Test Split (10:1 ratio) [cite: 195]
np.random.shuffle(valid_atoms)
split_idx = int(len(valid_atoms) * 10 / 11)

train_atoms = valid_atoms[:split_idx]
test_atoms = valid_atoms[split_idx:]

# 4. Write to Extended XYZ for MACE
write("cuox_train.extxyz", train_atoms)
write("cuox_test.extxyz", test_atoms)

```

### 2. The `mace_run_train` Command

The authors used a very specific set of hyperparameters to fine-tune the MACE-MP-0 foundation model. Because you are running on a Blackwell GPU (which has the same float32/TensorCore capabilities as the A100 used in the study ), you can run this command natively.

Make sure you have downloaded the pre-trained `MACE-MP-0` (medium) weights to your DGX before running this.

```bash
mace_run_train \
    --name="cuox_pbe_finetuned" \
    --foundation_model="path/to/2023-12-03-mace-mp-0-medium.model" \
    --train_file="cuox_train.extxyz" \
    --valid_file="cuox_test.extxyz" \
    --device="cuda" \
    --default_dtype="float32" \
    --batch_size=4 \
    --max_num_epochs=50 \
    --start_swa=51 \
    --lr=0.01 \
    --amsgrad=True \
    --ema=True \
    --ema_decay=0.99 \
    --E0s="average" \
    --energy_weight=1.0 \
    --forces_weight=1.0 

```

### Parameter Breakdown & Justification:

* 
`--foundation_model`: Points to the pre-trained MACE-MP-0 medium model.


* 
`--default_dtype="float32"`: Explicitly sets single-precision training, as performed on their A100 GPU.


* 
`--batch_size=4` & `--max_num_epochs=50`: Exact values from the SI.


* 
`--lr=0.01` & `--amsgrad=True`: The learning rate and the AMSGrad optimizer variant specified by the authors.


* 
`--ema=True` & `--ema_decay=0.99`: Implements the exponential moving average of model weights to stabilize training.


* 
`--E0s="average"`: Sets the atomic reference energies to the per-species average energies within your specific training set.


* 
`--energy_weight=1.0` & `--forces_weight=1.0`: Enforces the equal weighting of energy and force losses during single-head fine-tuning.


* `--start_swa=51`: We set Stochastic Weight Averaging (SWA) to start *after* the max epochs, effectively disabling it, as the authors relied on EMA rather than SWA for weight smoothing.

### 3. Verification

Once the training completes, `mace_run_train` will output validation metrics. You are aiming for metrics in the same neighborhood as the manuscript's GCGO-Finetuned test set errors:

* 
**Energy MAE:** ~ 9.8 meV/atom 


* 
**Force MAE:** ~ 35.3 meV/Å 



If your metrics land in this range, your fine-tuned MLIP is successfully localized to your PBE calculations and is ready to be plugged back into GOCIA for the massive Grand Canonical Genetic Algorithm (GCGA) minima searches.

Let me know when you've got the model trained, and we can map out the GCGA configuration parameters to handle the biased/unbiased chemical potential sweeps.