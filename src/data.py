"""Input loading, target/source label handling, and repository path layout."""

from pathlib import Path
from typing import Dict, List, Mapping, Set, Tuple

import numpy as np
import pandas as pd


def load_expression(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def load_sensitivity(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).iloc[:, :3].copy()
    df.columns = ["cell", "drug", "value"]
    df["cell"] = df["cell"].astype(str)
    df["drug"] = df["drug"].astype(str)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["value"])


def load_smiles(path: Path) -> Dict[str, str]:
    df = pd.read_csv(path).iloc[:, :2].copy()
    df.columns = ["drug", "smiles"]
    return {str(row.drug): str(row.smiles) for row in df.itertuples()}


def load_targets(path: Path) -> Dict[str, Set[str]]:
    df = pd.read_csv(path).iloc[:, :2].copy()
    df.columns = ["drug", "target"]
    out: Dict[str, Set[str]] = {}
    for row in df.itertuples():
        out.setdefault(str(row.drug), set()).add(str(row.target).upper())
    return out


def load_sc_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).iloc[:, :2].copy()
    df.columns = ["cell", "label"]
    df["cell"] = df["cell"].astype(str)
    df["label"] = df["label"].astype(int)
    return df


def normalize_genes(df: pd.DataFrame) -> pd.DataFrame:
    columns = df.columns.astype(str).str.upper().str.split(".").str[0].str.strip()
    out = df.copy()
    out.columns = columns
    return out.loc[:, ~out.columns.duplicated()]


def build_gdsc_labels(
    sensitivity: pd.DataFrame,
    top_frac: float,
    bottom_frac: float,
    higher_is_more_sensitive: bool,
    min_n: int,
) -> pd.DataFrame:
    """Construct binary sensitive/resistant labels from drug-wise response tails."""
    records: List[Tuple[str, str, int]] = []

    for drug, group in sensitivity.groupby("drug"):
        if len(group) < min_n:
            continue

        values = group["value"].values.astype(float)
        if not higher_is_more_sensitive:
            values = -values

        ranked = group.copy()
        ranked["_value"] = values
        ranked = ranked.sort_values("_value", ascending=False)

        n = len(ranked)
        n_top = max(1, round(n * top_frac))
        n_bottom = max(1, round(n * bottom_frac))

        records.extend((str(row.cell), drug, 1) for row in ranked.iloc[:n_top].itertuples())
        records.extend((str(row.cell), drug, 0) for row in ranked.iloc[-n_bottom:].itertuples())

    return pd.DataFrame(records, columns=["cell", "drug", "label"])


def resolve_paths(data_dir: Path) -> Dict[str, Path]:
    """Expected repository data layout for the single GSE112274 target task."""
    return {
        "gdsc_expr": data_dir / "gdsc" / "gdsc_expr.csv",
        "gdsc_sens": data_dir / "gdsc" / "gdsc_sens.csv",
        "gdsc_smiles": data_dir / "gdsc" / "gdsc_smiles.csv",
        "gdsc_targets": data_dir / "gdsc" / "gdsc_targets.csv",
        "target_expr": data_dir / "gse112274" / "GSE112274_expr.csv",
        "target_labels": data_dir / "gse112274" / "GSE112274_label.csv",
        "target_smiles": data_dir / "gse112274" / "target_smiles.csv",
        "target_targets": data_dir / "gse112274" / "target_targets.csv",
        "gmt": data_dir / "pathways" / "h.all.v2026.1.Hs.symbols.gmt",
    }


def validate_input_files(paths: Mapping[str, Path]) -> None:
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        pretty = "\n  - ".join(missing)
        raise FileNotFoundError(
            "Missing required input files:\n  - "
            + pretty
            + "\nSee data/README.md for the expected layout."
        )


def align_target_labels(target_expr: pd.DataFrame, labels_df: pd.DataFrame) -> np.ndarray:
    """Match GSE112274 labels to the expression matrix row order."""
    label_series = labels_df.set_index("cell")["label"]
    cell_ids = target_expr.index.astype(str)
    missing = [cell for cell in cell_ids if cell not in label_series.index]
    if missing:
        raise KeyError(
            f"{len(missing)} target cells are missing labels; first missing cell: {missing[0]}"
        )
    return label_series.loc[cell_ids].values.astype(int)
