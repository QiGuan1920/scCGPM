"""Rank-based pathway representation shared by bulk and single-cell RNA-seq."""

from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Set

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def load_gmt(path: Path) -> Dict[str, Set[str]]:
    gene_sets: Dict[str, Set[str]] = {}
    with path.open() as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                gene_sets[parts[0]] = {g.upper().strip() for g in parts[2:] if g}
    return gene_sets


def pathway_indices_for(
    df: pd.DataFrame, gmt: Mapping[str, Set[str]]
) -> Dict[str, List[int]]:
    position = {gene: i for i, gene in enumerate(df.columns)}
    return {
        name: [position[gene] for gene in genes if gene in position]
        for name, genes in gmt.items()
    }


def qualified_pathways(
    pathway_indices: Mapping[str, Sequence[int]], min_genes: int
) -> Set[str]:
    return {name for name, idx in pathway_indices.items() if len(idx) >= min_genes}


def compute_ranks(expr_values: np.ndarray) -> np.ndarray:
    """Compute intra-sample expression ranks, independently for each sample/cell."""
    return rankdata(expr_values, axis=1, method="average").astype(np.float32)


def pathway_scores(
    ranks: np.ndarray,
    pathway_indices: Mapping[str, Sequence[int]],
    pathway_names: Sequence[str],
    qualified: Set[str],
    fill: float = 0.5,
) -> np.ndarray:
    """Convert ranked expression into normalized pathway activity scores."""
    n_genes = ranks.shape[1]
    scores = np.full((ranks.shape[0], len(pathway_names)), fill, dtype=np.float32)

    for j, name in enumerate(pathway_names):
        if name not in qualified:
            continue
        idx = pathway_indices[name]
        m = len(idx)
        average_rank = ranks[:, idx].mean(axis=1)
        lower = (m + 1) / 2.0
        upper = n_genes - (m - 1) / 2.0
        scores[:, j] = (average_rank - lower) / max(upper - lower, 1e-8)

    return scores


def standardize_features(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return ((x - mean) / std).astype(np.float32)
