"""Multi-view drug representation and target-specific source-drug selection."""

from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, MACCSkeys, rdMolDescriptors

from .config import Config

RDLogger.DisableLog("rdApp.*")

DESC_NAMES = [
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "NumAromaticRings",
    "FractionCSP3",
    "FormalCharge",
]
MACCS_DIM = 167


def smiles_to_mol(smiles: str | None):
    if smiles is None or isinstance(smiles, float):
        return None
    return Chem.MolFromSmiles(str(smiles).strip())


def compute_maccs(mol) -> np.ndarray:
    if mol is None:
        return np.zeros(MACCS_DIM, dtype=np.float32)
    bit_vector = MACCSkeys.GenMACCSKeys(mol)
    array = np.zeros(MACCS_DIM, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bit_vector, array)
    return array


def compute_descriptors(mol) -> np.ndarray:
    if mol is None:
        return np.zeros(len(DESC_NAMES), dtype=np.float32)
    return np.array(
        [
            Descriptors.MolWt(mol),
            Crippen.MolLogP(mol),
            rdMolDescriptors.CalcTPSA(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol),
            Lipinski.NumRotatableBonds(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
            rdMolDescriptors.CalcFractionCSP3(mol),
            Chem.GetFormalCharge(mol),
        ],
        dtype=np.float32,
    )


class DrugInfo:
    def __init__(self, name: str, smiles: str, targets: Iterable[str]):
        self.name = name
        self.smiles = smiles
        self.targets = set(targets)
        molecule = smiles_to_mol(smiles)
        self.maccs = compute_maccs(molecule)
        self.desc_raw = compute_descriptors(molecule)
        self.desc = self.desc_raw.copy()
        self.mech: np.ndarray | None = None


def normalize_descriptors(
    drug_infos: Mapping[str, DrugInfo], source_drug_names: Sequence[str]
) -> None:
    source_desc = np.stack([drug_infos[drug].desc_raw for drug in source_drug_names])
    desc_min = source_desc.min(0)
    desc_max = source_desc.max(0)
    desc_range = desc_max - desc_min
    desc_range[desc_range < 1e-8] = 1.0

    for info in drug_infos.values():
        info.desc = (info.desc_raw - desc_min) / desc_range


def build_target_onehot(
    drug_infos: Mapping[str, DrugInfo], drug_names: Sequence[str]
) -> np.ndarray:
    targets = sorted({target for drug in drug_names for target in drug_infos[drug].targets})
    index = {target: i for i, target in enumerate(targets)}
    matrix = np.zeros((len(drug_names), len(targets)), dtype=np.float32)

    for i, drug in enumerate(drug_names):
        for target in drug_infos[drug].targets:
            matrix[i, index[target]] = 1.0

    return matrix


def set_mechanism_features(
    drug_infos: Mapping[str, DrugInfo],
    drug_names: Sequence[str],
    pathway_names: Sequence[str],
    gmt: Mapping[str, Set[str]],
    static_features: np.ndarray,
) -> np.ndarray:
    pathway_gene_sets = [gmt[name] for name in pathway_names]
    mechanism = np.zeros((len(drug_names), len(pathway_names)), dtype=np.float32)

    for i, drug in enumerate(drug_names):
        targets = drug_infos[drug].targets
        if targets:
            for j, gene_set in enumerate(pathway_gene_sets):
                if targets & gene_set:
                    mechanism[i, j] = 1.0
        drug_infos[drug].mech = mechanism[i]

    return np.concatenate([static_features, mechanism], axis=1).astype(np.float32)


def tanimoto_bits(a: np.ndarray, b: np.ndarray) -> float:
    intersection = float(np.sum((a > 0) & (b > 0)))
    union = float(np.sum((a > 0) | (b > 0)))
    return intersection / union if union > 0 else 0.0


def jaccard_sets(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union > 0 else 0.0


def cosine_similarity_np(a: np.ndarray, b: np.ndarray) -> float:
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (norm_a * norm_b)) if norm_a > 0 and norm_b > 0 else 0.0


def compute_source_weights(
    target_info: DrugInfo,
    drug_infos: Mapping[str, DrugInfo],
    source_names: Sequence[str],
    cfg: Config,
) -> Tuple[Dict[str, float], List[Dict[str, float]], Set[str]]:
    """Fuse structure, target, and pathway similarity into transfer weights."""
    similarities: List[float] = []
    report: List[Dict[str, float]] = []

    if target_info.mech is None:
        raise RuntimeError("Target mechanism features must be initialized first.")

    for drug in source_names:
        source_info = drug_infos[drug]
        if source_info.mech is None:
            raise RuntimeError(f"Mechanism features not initialized for {drug}.")

        structure = tanimoto_bits(source_info.maccs, target_info.maccs)
        target = jaccard_sets(source_info.targets, target_info.targets)
        pathway = cosine_similarity_np(source_info.mech, target_info.mech)
        combined = cfg.alpha * structure + cfg.beta * target + cfg.gamma * pathway

        similarities.append(combined)
        report.append(
            {
                "drug": drug,
                "structure_similarity": structure,
                "target_similarity": target,
                "pathway_similarity": pathway,
                "combined_similarity": combined,
            }
        )

    similarities_array = np.asarray(similarities, dtype=np.float32)
    mask = np.ones_like(similarities_array, dtype=bool)

    if cfg.top_k_source_drugs < len(similarities_array):
        mask[:] = False
        mask[np.argsort(-similarities_array)[: cfg.top_k_source_drugs]] = True

    logits = np.where(mask, similarities_array / cfg.tau, -1e9)
    logits -= logits.max()
    weights = np.exp(logits)
    weights /= weights.sum()

    omega = {source_names[i]: float(weights[i]) for i in range(len(weights))}
    selected = {source_names[i] for i in range(len(source_names)) if mask[i]}

    for row in report:
        row["transfer_weight"] = omega[row["drug"]]
        row["selected"] = row["drug"] in selected

    return omega, report, selected
