"""End-to-end orchestration for the single GSE112274 / Gefitinib experiment."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Set

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD

from .config import Config
from .data import (
    align_target_labels,
    build_gdsc_labels,
    load_expression,
    load_sc_labels,
    load_sensitivity,
    load_smiles,
    load_targets,
    normalize_genes,
    resolve_paths,
    validate_input_files,
)
from .drug import (
    DrugInfo,
    build_target_onehot,
    compute_source_weights,
    normalize_descriptors,
    set_mechanism_features,
    smiles_to_mol,
)
from .evaluate import evaluate_target
from .pathway import (
    compute_ranks,
    load_gmt,
    pathway_indices_for,
    pathway_scores,
    qualified_pathways,
    standardize_features,
)
from .train import adapt_to_target, pretrain_source
from .utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="scCGPM: GSE112274 / Gefitinib cross-domain response prediction."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Repository data directory (default: ./data).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/gse112274"),
        help="Output directory (default: ./results/gse112274).",
    )
    return parser.parse_args()


def _build_drug_representation(
    cfg: Config,
    gdsc_smiles: Dict[str, str],
    gdsc_targets: Dict[str, Set[str]],
    target_smiles_map: Dict[str, str],
    target_targets_map: Dict[str, Set[str]],
    source_drug_names: list[str],
    pathway_names: list[str],
    gmt: Dict[str, Set[str]],
):
    smiles_map: Dict[str, str] = {
        drug: gdsc_smiles[drug] for drug in source_drug_names
    }
    target_map: Dict[str, Set[str]] = {
        drug: gdsc_targets.get(drug, set()) for drug in source_drug_names
    }

    if cfg.target_drug not in smiles_map:
        if cfg.target_drug not in target_smiles_map:
            raise KeyError(f"No SMILES found for target drug: {cfg.target_drug}")
        smiles_map[cfg.target_drug] = target_smiles_map[cfg.target_drug]
        target_map[cfg.target_drug] = target_targets_map.get(cfg.target_drug, set())

    all_drug_names = sorted(smiles_map)
    drug_to_idx = {drug: i for i, drug in enumerate(all_drug_names)}
    idx_to_drug = {i: drug for drug, i in drug_to_idx.items()}

    drug_infos = {
        drug: DrugInfo(drug, smiles_map[drug], target_map.get(drug, set()))
        for drug in all_drug_names
    }

    if smiles_to_mol(drug_infos[cfg.target_drug].smiles) is None:
        raise ValueError(f"Invalid SMILES for target drug: {cfg.target_drug}")

    normalize_descriptors(drug_infos, source_drug_names)

    target_onehot = build_target_onehot(drug_infos, all_drug_names)
    if target_onehot.shape[1] < 2:
        raise ValueError(
            "At least two distinct drug targets are required for the target-SVD representation."
        )

    source_indices = [drug_to_idx[drug] for drug in source_drug_names]
    n_components = min(cfg.target_svd_dim, target_onehot.shape[1] - 1)
    target_svd = TruncatedSVD(n_components=n_components, random_state=cfg.seed)
    target_svd.fit(target_onehot[source_indices])
    target_svd_features = target_svd.transform(target_onehot).astype(np.float32)

    static_features = np.stack(
        [
            np.concatenate(
                [drug_infos[drug].maccs, drug_infos[drug].desc, target_svd_features[i]]
            )
            for i, drug in enumerate(all_drug_names)
        ]
    ).astype(np.float32)

    drug_features = set_mechanism_features(
        drug_infos,
        all_drug_names,
        pathway_names,
        gmt,
        static_features,
    )

    return (
        drug_infos,
        all_drug_names,
        drug_to_idx,
        idx_to_drug,
        drug_features,
    )


def run_experiment(data_dir: Path, output_dir: Path) -> None:
    cfg = Config()
    set_seed(cfg.seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = resolve_paths(data_dir)
    validate_input_files(paths)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"target: {cfg.target_dataset} | {cfg.target_drug}")

    # 1) Load one source domain (GDSC) and one target domain (GSE112274).
    gdsc_expr = normalize_genes(load_expression(paths["gdsc_expr"]))
    gdsc_sensitivity = load_sensitivity(paths["gdsc_sens"])
    gdsc_smiles = load_smiles(paths["gdsc_smiles"])
    gdsc_targets = load_targets(paths["gdsc_targets"])

    target_expr = normalize_genes(load_expression(paths["target_expr"]))
    target_labels_df = load_sc_labels(paths["target_labels"])
    target_smiles_map = load_smiles(paths["target_smiles"])
    target_targets_map = load_targets(paths["target_targets"])

    gmt = load_gmt(paths["gmt"])
    print(f"GMT pathways: {len(gmt)}")

    # 2) Build the shared intra-sample rank-based pathway space.
    gdsc_pw_idx = pathway_indices_for(gdsc_expr, gmt)
    target_pw_idx = pathway_indices_for(target_expr, gmt)
    gdsc_qualified = qualified_pathways(gdsc_pw_idx, cfg.min_genes_per_pathway)
    target_qualified = qualified_pathways(target_pw_idx, cfg.min_genes_per_pathway)
    pathway_names = sorted(gdsc_qualified | target_qualified)

    print(
        f"qualified pathway union: {len(pathway_names)} "
        f"(GDSC-only={len(gdsc_qualified - target_qualified)}, "
        f"target-only={len(target_qualified - gdsc_qualified)}, "
        f"shared={len(gdsc_qualified & target_qualified)})"
    )

    source_pathways_full = pathway_scores(
        compute_ranks(gdsc_expr.values.astype(np.float32)),
        gdsc_pw_idx,
        pathway_names,
        gdsc_qualified,
        fill=0.5,
    )
    target_pathways = pathway_scores(
        compute_ranks(target_expr.values.astype(np.float32)),
        target_pw_idx,
        pathway_names,
        target_qualified,
        fill=0.5,
    )

    # 3) Construct GDSC source labels.
    gdsc_labels = build_gdsc_labels(
        gdsc_sensitivity,
        cfg.top_frac,
        cfg.bottom_frac,
        cfg.higher_is_more_sensitive,
        cfg.min_samples_per_drug,
    )
    gdsc_labels = gdsc_labels[
        gdsc_labels["cell"].isin(set(gdsc_expr.index.astype(str)))
        & gdsc_labels["drug"].isin(gdsc_smiles.keys())
    ]
    source_drug_names = sorted(gdsc_labels["drug"].unique())

    cell_to_row = {cell: i for i, cell in enumerate(gdsc_expr.index.astype(str))}
    source_rows = np.array(
        [cell_to_row[row.cell] for row in gdsc_labels.itertuples()], dtype=np.int64
    )
    source_y_all = np.array(
        [int(row.label) for row in gdsc_labels.itertuples()], dtype=np.int64
    )

    # 4) Construct the multi-view drug representation.
    (
        drug_infos,
        all_drug_names,
        drug_to_idx,
        idx_to_drug,
        drug_features,
    ) = _build_drug_representation(
        cfg,
        gdsc_smiles,
        gdsc_targets,
        target_smiles_map,
        target_targets_map,
        source_drug_names,
        pathway_names,
        gmt,
    )
    drug_features_t = torch.tensor(
        drug_features, dtype=torch.float32, device=device
    )
    source_drug_idx_all = np.array(
        [drug_to_idx[row.drug] for row in gdsc_labels.itertuples()], dtype=np.int64
    )

    # 5) Select/weight GDSC drugs specifically for Gefitinib.
    omega, similarity_report, selected_source_drugs = compute_source_weights(
        drug_infos[cfg.target_drug], drug_infos, source_drug_names, cfg
    )

    selected_indices = {drug_to_idx[drug] for drug in selected_source_drugs}
    sample_mask = np.array(
        [
            i
            for i, drug_idx in enumerate(source_drug_idx_all)
            if drug_idx in selected_indices
        ],
        dtype=np.int64,
    )

    source_pathways = standardize_features(source_pathways_full[source_rows])[sample_mask]
    target_pathways = standardize_features(target_pathways)
    source_drug_idx = source_drug_idx_all[sample_mask]
    source_y = source_y_all[sample_mask]
    source_sample_weights = np.array(
        [omega.get(idx_to_drug[idx], 0.0) for idx in source_drug_idx],
        dtype=np.float32,
    )

    print(f"selected source drugs: {', '.join(sorted(selected_source_drugs))}")
    print(f"selected source samples: {len(source_y)}/{len(source_y_all)}")

    target_y = align_target_labels(target_expr, target_labels_df)

    # 6) Source pretraining -> GSE112274 adaptation.
    model, source_val_auc, _ = pretrain_source(
        source_pathways,
        source_drug_idx,
        source_y,
        drug_features_t,
        n_pathways=len(pathway_names),
        drug_feat_dim=drug_features.shape[1],
        cfg=cfg,
        device=device,
    )

    model = adapt_to_target(
        model,
        source_pathways,
        source_drug_idx,
        source_y,
        source_sample_weights,
        target_pathways,
        target_drug_idx=drug_to_idx[cfg.target_drug],
        drug_features_t=drug_features_t,
        cfg=cfg,
        device=device,
    )

    # 7) Evaluation and reproducible outputs.
    metrics, probability, prediction, gates, threshold = evaluate_target(
        model,
        target_pathways,
        target_drug_idx=drug_to_idx[cfg.target_drug],
        target_labels=target_y,
        drug_features_t=drug_features_t,
        device=device,
    )
    metrics["source_val_AUC"] = float(source_val_auc)

    metrics_df = pd.DataFrame([metrics], index=[cfg.target_dataset])
    metrics_df.index.name = "dataset"
    metrics_df.to_csv(output_dir / "metrics.csv")

    pd.DataFrame(
        {
            "cell": target_expr.index.astype(str),
            "drug": cfg.target_drug,
            "score": probability,
            "pred": prediction,
            "true": target_y,
            "threshold": threshold,
        }
    ).to_csv(output_dir / "predictions.csv", index=False)

    pd.DataFrame(similarity_report).sort_values(
        ["selected", "transfer_weight", "combined_similarity"],
        ascending=[False, False, False],
    ).to_csv(output_dir / "source_drug_weights.csv", index=False)

    gate_df = pd.DataFrame(
        gates, columns=pathway_names, index=target_expr.index.astype(str)
    )
    gate_df.index.name = "cell"
    gate_df.to_csv(output_dir / "target_pathway_gates.csv")

    pd.Series(pathway_names, name="pathway").to_csv(
        output_dir / "pathways.csv", index=False
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "pathways": pathway_names,
            "drug_names": all_drug_names,
        },
        output_dir / "model.pt",
    )

    with (output_dir / "run_config.json").open("w") as handle:
        json.dump(asdict(cfg), handle, indent=2)

    print("\nGSE112274 evaluation")
    print(metrics_df.round(4).to_string())
    print(f"\nOutputs saved to: {output_dir}")


def main() -> None:
    args = parse_args()
    run_experiment(args.data_dir.resolve(), args.output_dir.resolve())
