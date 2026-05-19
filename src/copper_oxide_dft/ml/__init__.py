"""MLIP-GCGO pipeline (machine-learned interatomic potential + grand-canonical
genetic algorithm) for the non-aqueous CuO/Cu(111) structural-discovery
question.

See :mod:`docs/ml-gcgo-pivot.md` for the scientific scope and decisions; this
package implements the workflow.

Modules:

- :mod:`box_sampling` — perturb seed bulks (rattle, isotropic scale,
  random O insert/delete) with Hookean pre-optimization and Cu-O
  connectivity filtering. Produces the structural prior for the DFT
  ground-truth dataset.
- :mod:`qe_driver` — wrap ASE-Espresso with the converged Phase 1
  settings; batched relaxations on DGX Spark.
- :mod:`curate` — force-filter, SOAP+IPCA+UMAP subsample, extxyz writers.
- :mod:`validate` — held-out test-set MAE for the fine-tuned MACE model.
- :mod:`gcga` — GOCIA-driven GCGA on Cu(111).
- :mod:`ensemble` — per-x_O minimum-Ω extraction.
- :mod:`fcp_rerank` — Frontier ESM-FCP rerank inputs + ranking.
- :mod:`sld` — neutron-reflectometry SLD calculator.

This ``__init__`` re-exports only the *public entry points* and result
dataclasses. Internal helpers (``compute_x_o``, ``enforce_cu_o_connectivity``,
``filter_by_max_force``, ``energy_mae_per_atom_mev`` etc.) live in their
submodules — import them directly from there when needed. Trimming the
top-level surface keeps the package layout free to evolve without
breaking imports across the codebase.
"""

from copper_oxide_dft.ml.box_sampling import (
    BoxSamplingConfig,
    PerturbationResult,
    sample_batch,
)
from copper_oxide_dft.ml.curate import (
    DatasetSplit,
    prepare_dataset,
)
from copper_oxide_dft.ml.ensemble import (
    Phase,
    per_x_o_minima,
    read_ensemble_extxyz,
    top_k_by_omega,
    write_ensemble_extxyz,
)
from copper_oxide_dft.ml.fcp_rerank import (
    DEFAULT_AG_AGCL_ABSOLUTE_POTENTIAL_V,
    DEFAULT_TARGET_POTENTIAL_V,
    FcpRerankResult,
    prepare_fcp_inputs,
    rank_fcp_results,
    write_frontier_submit_scripts,
)
from copper_oxide_dft.ml.gcga import (
    GCGAConfig,
    build_cu111_gcga_substrate,
)
from copper_oxide_dft.ml.qe_driver import (
    DatasetEntry,
    read_dataset_outputs,
    write_dataset_inputs,
)
from copper_oxide_dft.ml.sld import (
    SldProfile,
    bulk_cu_normalization_factor,
    compute_sld_profile,
)
from copper_oxide_dft.ml.validate import (
    ValidationMetrics,
    evaluate_model_on_extxyz,
)

__all__ = [
    # Result + config dataclasses
    "BoxSamplingConfig",
    "DatasetEntry",
    "DatasetSplit",
    "FcpRerankResult",
    "GCGAConfig",
    "PerturbationResult",
    "Phase",
    "SldProfile",
    "ValidationMetrics",
    # Public constants
    "DEFAULT_AG_AGCL_ABSOLUTE_POTENTIAL_V",
    "DEFAULT_TARGET_POTENTIAL_V",
    # Public entry points
    "build_cu111_gcga_substrate",
    "bulk_cu_normalization_factor",
    "compute_sld_profile",
    "evaluate_model_on_extxyz",
    "per_x_o_minima",
    "prepare_dataset",
    "prepare_fcp_inputs",
    "rank_fcp_results",
    "read_dataset_outputs",
    "read_ensemble_extxyz",
    "sample_batch",
    "top_k_by_omega",
    "write_dataset_inputs",
    "write_ensemble_extxyz",
    "write_frontier_submit_scripts",
]
