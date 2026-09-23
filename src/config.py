"""Configuration for the focused GSE112274 / Gefitinib experiment."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Reproducibility / runtime
    seed: int = 1

    # Fixed target-domain task in this release
    target_dataset: str = "GSE112274"
    target_drug: str = "Gefitinib"

    # Source label construction
    top_frac: float = 0.10
    bottom_frac: float = 0.10
    higher_is_more_sensitive: bool = True
    min_samples_per_drug: int = 20

    # Pathway / drug representation
    min_genes_per_pathway: int = 5
    target_svd_dim: int = 100

    # Network
    adapter_hidden: int = 128
    z_dim: int = 64
    drug_out: int = 32
    dropout: float = 0.5

    # Multi-view source-drug similarity
    alpha: float = 0.50      # structure
    beta: float = 0.25       # target
    gamma: float = 0.25      # pathway mechanism
    tau: float = 0.50
    top_k_source_drugs: int = 3

    # Domain adaptation
    lambda_align: float = 20.0
    lambda_manifold: float = 0.05
    manifold_k: int = 10

    # Pseudo labels
    min_cells_per_class: int = 10
    pseudo_update_every: int = 3
    pseudo_quantile: float = 0.20

    # Training
    source_epochs: int = 15
    source_patience: int = 6
    adapt_epochs: int = 10
    refine_epochs: int = 10
    batch_size: int = 128
    target_batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-2
