# copper-oxide-dft

DFT pipeline for copper oxide phases on Cu(111) under applied electrochemical potential, using **Quantum ESPRESSO**. The end goal is potential-driven surface reconstruction in aqueous solution; intermediate phases produce a Pourbaix-style stability diagram and validated bulk references.

**Status:** Early development. Phase 0 (toolchain) is complete; Phase 1 (bulk-Cu convergence) is the next scientific milestone. See [docs/implementation-plan.md](docs/implementation-plan.md) for the full 9-phase roadmap and [docs/ground_truths.md](docs/ground_truths.md) for the locked methodology decisions and Cu-oxide DFT gotchas.

## What this package gives you

- Python builders for QE input structures (currently bulk fcc Cu).
- A `pw.x` input file generator with project-standard defaults: PBE-ready namelists, Marzari-Vanderbilt smearing, `ibrav=0`, `conv_thr=1e-8`, sensible PAW cutoffs.
- Convergence-sweep helpers for `ecutwfc`, k-points, and smearing.
- SLURM submission scripts targeted at **ORNL Frontier** (AMD MI250X GPUs, 8 GCDs/node, GPU-aware MPICH) with an **Andes** (CPU) preset for debugging.
- A parser for the converged `pw.x` output (total energy, Fermi energy, magnetization, JOB DONE).
- An `inspect` command to verify any generated input structurally (cell, composition, layer-by-layer atoms) before committing compute time.

## Install

```bash
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
```

Requires Python 3.10+.

## One-time setup: pseudopotentials

Download a Cu pseudopotential from [PseudoDojo](http://www.pseudo-dojo.org/) (PBE, scalar-relativistic, standard accuracy, PAW), drop it into a directory, and point an env var at it:

```bash
mkdir -p ~/pseudos
mv Cu.upf ~/pseudos/
export CUOXDFT_PSEUDO_DIR=~/pseudos
```

Add the `export` line to your `~/.bashrc` / `~/.zshrc` so it persists. O and H pseudopotentials get added when Phase 2 (oxide bulks) lands.

## Quickstart

### Generate a single pw.x input and verify it

```bash
copper-oxide-dft bulk-cu --out runs/bulk_cu/pw.in
copper-oxide-dft inspect runs/bulk_cu/pw.in
```

`inspect` decodes the input back into an ASE structure and prints cell vectors, composition, and a layer-by-layer atom breakdown — the same view will make slab/oxide structures (Phase 3 onward) visually verifiable before you spend cluster hours on them.

### Phase 1 convergence sweep -> Frontier

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

# 4) Pull results back, parse.
rsync -av frontier:scratch/conv_ecutwfc/ runs/conv_ecutwfc/
copper-oxide-dft parse --json runs/conv_ecutwfc/*/pw.out
```

### Local workstation (Ubuntu, CPU)

See [docs/local-workstation.md](docs/local-workstation.md) for a full end-to-end walkthrough on a Linux box (including a note on AMD consumer GPUs + Q-E).

## CLI reference

| Command | Purpose |
|---|---|
| `bulk-cu` | Generate a bulk-Cu pw.x input (scf / relax / vc-relax). |
| `sweep` | Generate a tree of pw.x inputs varying `ecutwfc`, `kpts`, or `degauss`. |
| `inspect` | Decode a pw.x input and print cell, composition, layer-by-layer atoms. |
| `make-slurm` | Emit submit.sh next to each pw.in (target: Frontier or Andes). |
| `parse` | Read pw.x stdout files for total energy, Fermi energy, magnetization, JOB DONE. |

Every command has `--help`.

## Project layout

```
src/copper_oxide_dft/
├── structure_builder.py    # ASE constructors + layer summaries
├── qe_input.py             # pw.x input generation (scf/relax/vc-relax)
├── convergence.py          # ecutwfc / kpts / degauss sweep helpers
├── parse.py                # pw.x stdout parsing
├── submit.py               # SLURM scaffolding (Frontier, Andes)
└── cli.py                  # Click CLI

docs/
├── project.md              # scope and dependencies
├── implementation-plan.md  # 9-phase roadmap (current focus: Phase 1)
├── ground_truths.md        # methodology decisions, Cu-oxide DFT gotchas, Frontier conventions
└── local-workstation.md    # Ubuntu walkthrough
```

## Development

```bash
pytest                     # all tests with coverage
ruff check src tests       # lint
ruff format src tests      # format
```

CI-ready: ~50 tests, ~99% coverage on `src/copper_oxide_dft/`.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
