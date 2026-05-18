# copper-oxide-dft

DFT pipeline for copper oxide phases on Cu(111) under applied electrochemical potential, built on **Quantum ESPRESSO**. The end goal is potential-driven surface reconstruction; intermediate milestones produce a Pourbaix-style stability diagram and validated bulk references.

**Status:** Python scaffolding for Phases 0–8 is complete (165 tests, ~97 % coverage). Production HPC runs (Phases 1–3 onward) are next. See [docs/implementation-plan.md](docs/implementation-plan.md) for the 9-phase roadmap and [docs/ground_truths.md](docs/ground_truths.md) for locked methodology decisions and Cu-oxide DFT gotchas.

## Pick your starting point

| If you want to… | Read this |
|---|---|
| Run **CuO on Cu(111) at −0.8 V in a non-aqueous electrolyte**, prototype on DGX Spark (GB10), then move to Frontier | [docs/startup-cuo-cu-nonaqueous.md](docs/startup-cuo-cu-nonaqueous.md) |
| Validate the **full aqueous pipeline (Phases 1–8) on an Ubuntu CPU workstation** before any cluster hours | [docs/local-workstation.md](docs/local-workstation.md) |
| Understand the **scientific roadmap** and the rationale for each phase | [docs/implementation-plan.md](docs/implementation-plan.md) |
| Look up an **API quirk, AFM/Hubbard-U detail, or Frontier convention** that bit us before | [docs/ground_truths.md](docs/ground_truths.md) |

## What this package gives you

- **Structure builders** (ASE-based): bulk Cu / Cu₂O / CuO with correct AFM ordering; Cu(111) / Cu₂O(111) / CuO(111) slabs; O / OH adsorbates at arbitrary coverage; explicit-water layers; H₂ / H₂O / O₂ reference molecules.
- **`pw.x` input generation** with project defaults: PBE-ready namelists, Marzari–Vanderbilt smearing, `ibrav=0`, `conv_thr=1e-8`, PAW cutoffs, automatic species-splitting for AFM with Hubbard U on both sublattices.
- **Convergence sweeps** over `ecutwfc`, k-points, smearing, and Hubbard U; analyser that picks the smallest converged value to within a per-atom threshold.
- **`hp.x`** input writer for self-consistent Hubbard-U linear response.
- **CHE post-processing** (Computational Hydrogen Electrode): bulk Pourbaix for Cu / Cu₂O / CuO and adsorbate Pourbaix for O/OH on Cu(111). Aqueous-only — see the non-aqueous startup guide for the non-aqueous route.
- **Environ** implicit-solvation input writer (default water, override for non-aqueous).
- **ESM-FCP** constant-potential helper: maps U vs. SHE (or any reference) into the right `lfcp` / `assume_isolated='esm'` / `fcp_mu` triple.
- **NEB** input writer for `neb.x` reconstruction-barrier studies.
- **SLURM submission** scripts for ORNL Frontier (AMD MI250X, GPU-aware MPICH) and Andes (CPU).
- **Output parsing**: total energy, Fermi energy, magnetization, JOB DONE.
- **`inspect` command** to decode a generated input and print cell + composition + layer-by-layer atoms — the cheapest catch for malformed structures.

## Install

```bash
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
```

Requires Python 3.10+.

## One-time setup: pseudopotentials

Download Cu, O, H pseudopotentials from [PseudoDojo](http://www.pseudo-dojo.org/) (PBE, scalar-relativistic, standard accuracy, PAW), drop them into a directory, and point an env var at it:

```bash
mkdir -p ~/pseudos
mv {Cu,O,H}.upf ~/pseudos/
export CUOXDFT_PSEUDO_DIR=~/pseudos
```

Add the `export` line to your `~/.bashrc` / `~/.zshrc` so it persists.

## Quickstart: 60-second sanity check

```bash
# 1. Build a single bulk-Cu input and verify it.
copper-oxide-dft bulk-cu --out runs/bulk_cu/pw.in
copper-oxide-dft inspect runs/bulk_cu/pw.in

# 2. Run it locally (assuming pw.x is on PATH).
(cd runs/bulk_cu && pw.x -in pw.in > pw.out)

# 3. Parse the result.
copper-oxide-dft parse runs/bulk_cu/pw.out   # expect done=True
```

For a real workflow (convergence sweeps → bulk oxides → surfaces → electrochemistry), pick the matching guide from the table above.

## Quickstart: Phase 1 convergence sweep → Frontier

```bash
# 1) Generate a sweep tree (local).
copper-oxide-dft sweep \
  --param ecutwfc --values 40,60,80,100 \
  --out runs/conv_ecutwfc

# 2) Emit a submit.sh next to every pw.in (defaults target Frontier).
copper-oxide-dft make-slurm runs/conv_ecutwfc \
  --account <your-project> \
  --qe-module quantum-espresso/<version>-gpu \
  --walltime 0:30:00

# 3) Ship to Frontier and submit.
rsync -av runs/conv_ecutwfc/ frontier:scratch/conv_ecutwfc/
ssh frontier
for d in scratch/conv_ecutwfc/*/; do (cd "$d" && sbatch submit.sh); done

# 4) Pull results back, analyse.
rsync -av frontier:scratch/conv_ecutwfc/ runs/conv_ecutwfc/
copper-oxide-dft sweep-analyze runs/conv_ecutwfc --threshold-mev 1 \
  --png runs/conv_ecutwfc/convergence.png
```

## Quickstart: Pourbaix end-to-end (aqueous)

```bash
# 1) Bundle the five DFT inputs needed for the CHE Pourbaix.
copper-oxide-dft make-pourbaix-inputs runs/phase4

# 2) Wrap them for Frontier (or run locally — see local-workstation.md).
copper-oxide-dft make-slurm runs/phase4 --account <your-project> --walltime 2:00:00
# (ship + submit + wait)

# 3) Aggregate the pw.out energies into the schema the pourbaix CLI consumes.
copper-oxide-dft aggregate-pourbaix-energies runs/phase4 \
  --out runs/phase4/energies.json

# 4) Build the diagram, mark a (U, pH) point, save a PNG.
copper-oxide-dft pourbaix \
  --u -0.4 --ph 7 \
  --energies runs/phase4/energies.json \
  --png runs/phase4/pourbaix.png
```

Without `--energies`, the CLI falls back to literature ΔG_f values so the diagram is qualitatively correct out of the box — handy for testing the plotting path before any DFT lands.

## CLI reference

| Command | Purpose |
|---|---|
| `bulk-cu` | Generate a bulk-Cu `pw.x` input (`scf` / `relax` / `vc-relax`). |
| `sweep` | Generate a tree of `pw.x` inputs varying `ecutwfc`, `kpts`, `degauss`, or `hubbard_u`. |
| `sweep-analyze` | Parse a sweep tree, print a per-point table, pick the smallest converged value, optional PNG. |
| `inspect` | Decode a `pw.x` input and print cell, composition, layer-by-layer atoms. |
| `make-slurm` | Emit `submit.sh` next to each `pw.in` (target: Frontier or Andes). |
| `parse` | Read `pw.x` stdout files for total energy, Fermi energy, magnetization, JOB DONE. |
| `make-pourbaix-inputs` | Generate the 5 inputs (Cu, Cu₂O, CuO, H₂, H₂O) for a CHE Pourbaix. |
| `aggregate-pourbaix-energies` | Parse a `make-pourbaix-inputs` tree into the JSON shape the `pourbaix` CLI consumes. |
| `pourbaix` | Build a Cu / Cu₂O / CuO Pourbaix diagram, mark a point, save PNG / JSON. |

Every command has `--help`.

## Project layout

```
src/copper_oxide_dft/
├── structure_builder.py    # ASE constructors + layer summaries
├── qe_input.py             # pw.x input generation, AFM/Hubbard helper, hp.x writer, FCP helper
├── convergence.py          # ecutwfc / kpts / degauss / hubbard_u sweep helpers
├── analysis.py             # sweep-analyze: convergence picker + plots
├── parse.py                # pw.x stdout parsing
├── submit.py               # SLURM scaffolding (Frontier, Andes)
├── che.py                  # Computational Hydrogen Electrode (bulk + adsorbate)
├── pourbaix.py             # Pourbaix diagram construction + plotting
├── environ.py              # environ.in writer (implicit solvation)
├── neb.py                  # neb.x input writer
├── config.py               # JSON-backed converged-parameter store
└── cli.py                  # Click CLI

docs/
├── project.md              # scope and dependencies
├── implementation-plan.md  # 9-phase roadmap (aqueous default)
├── ground_truths.md        # methodology decisions, Cu-oxide DFT gotchas, Frontier conventions
├── local-workstation.md    # Ubuntu CPU walkthrough (Phases 1–8, aqueous)
└── startup-cuo-cu-nonaqueous.md   # DGX Spark → Frontier walkthrough (non-aqueous, −0.8 V)
```

## Development

```bash
pytest                     # all tests with coverage (~165 tests)
ruff check src tests       # lint
ruff format src tests      # format
```

## License

BSD-3-Clause. See [LICENSE](LICENSE).
