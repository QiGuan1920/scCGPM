"""Source pretraining and pseudo-label-guided target adaptation."""

import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from .config import Config
from .losses import coral_loss, manifold_loss
from .model import ChemGuideModel
from .utils import seeded_generator, set_seed


def make_pseudo_labels(
    model: ChemGuideModel,
    target_x: torch.Tensor,
    target_feat: torch.Tensor,
    cfg: Config,
):
    """Use high-/low-confidence prediction quantiles as target pseudo labels."""
    model.eval()
    with torch.no_grad():
        latent, _ = model.encode(target_x, target_feat.expand(target_x.shape[0], -1))
        probability = torch.sigmoid(model.logits(latent)).cpu().numpy()

    q = cfg.pseudo_quantile
    high, low = np.quantile(probability, 1 - q), np.quantile(probability, q)
    idx_pos = np.where(probability >= high)[0]
    idx_neg = np.where(probability <= low)[0]

    if len(idx_neg) < cfg.min_cells_per_class or len(idx_pos) < cfg.min_cells_per_class:
        return None

    return (
        torch.tensor(idx_neg, dtype=torch.long, device=target_x.device),
        torch.tensor(idx_pos, dtype=torch.long, device=target_x.device),
    )


def class_conditional_alignment(
    model: ChemGuideModel,
    source_z: torch.Tensor,
    source_y: torch.Tensor,
    target_x: torch.Tensor,
    target_feat: torch.Tensor,
    target_batch_indices: torch.Tensor,
    pseudo,
    source_weights: torch.Tensor | None,
):
    """Align sensitive/resistant subspaces using similarity-weighted CORAL."""
    idx_neg_all, idx_pos_all = pseudo
    batch_members = set(target_batch_indices.tolist())

    idx_neg = torch.tensor(
        [idx for idx in idx_neg_all.tolist() if idx in batch_members],
        dtype=torch.long,
        device=target_x.device,
    )
    idx_pos = torch.tensor(
        [idx for idx in idx_pos_all.tolist() if idx in batch_members],
        dtype=torch.long,
        device=target_x.device,
    )

    source_neg = source_z[source_y == 0]
    source_pos = source_z[source_y == 1]
    weight_neg = source_weights[source_y == 0] if source_weights is not None else None
    weight_pos = source_weights[source_y == 1] if source_weights is not None else None

    loss = None
    if idx_neg.numel() >= 2 and source_neg.shape[0] >= 2:
        target_neg_z, _ = model.encode(
            target_x[idx_neg], target_feat.expand(idx_neg.numel(), -1)
        )
        loss = coral_loss(source_neg, target_neg_z, weight_neg)

    if idx_pos.numel() >= 2 and source_pos.shape[0] >= 2:
        target_pos_z, _ = model.encode(
            target_x[idx_pos], target_feat.expand(idx_pos.numel(), -1)
        )
        term = coral_loss(source_pos, target_pos_z, weight_pos)
        loss = term if loss is None else loss + term

    return loss


def pretrain_source(
    source_pathways: np.ndarray,
    source_drug_idx: np.ndarray,
    source_y: np.ndarray,
    drug_features_t: torch.Tensor,
    n_pathways: int,
    drug_feat_dim: int,
    cfg: Config,
    device: torch.device,
):
    """Supervised source pretraining on target-relevant GDSC drugs."""
    set_seed(cfg.seed)
    model = ChemGuideModel(n_pathways, drug_feat_dim, cfg).to(device)

    idx_all = np.arange(len(source_y))
    x_train, x_val, d_train, d_val, y_train, y_val, _, idx_val = train_test_split(
        source_pathways,
        source_drug_idx,
        source_y,
        idx_all,
        test_size=0.2,
        random_state=cfg.seed,
        stratify=source_y,
    )

    if (y_train == 1).sum() and (y_train == 0).sum():
        proto_pos = x_train[y_train == 1].mean(0).astype(np.float32)
        proto_neg = x_train[y_train == 0].mean(0).astype(np.float32)
    else:
        proto_pos = np.zeros(n_pathways, dtype=np.float32)
        proto_neg = np.zeros(n_pathways, dtype=np.float32)
    model.set_prototypes(proto_pos, proto_neg)

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_train),
            torch.tensor(d_train),
            torch.tensor(y_train, dtype=torch.float32),
        ),
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=seeded_generator(cfg.seed),
    )

    x_val_t = torch.tensor(x_val, dtype=torch.float32, device=device)
    d_val_t = torch.tensor(d_val, dtype=torch.long, device=device)

    pos_weight = torch.tensor(
        [(y_train == 0).sum() / max((y_train == 1).sum(), 1)],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    best_auc = -1.0
    best_state = None
    patience = 0

    for epoch in range(cfg.source_epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for pathways_b, drug_b, y_b in train_loader:
            pathways_b = pathways_b.to(device)
            drug_b = drug_b.to(device)
            y_b = y_b.to(device)

            latent, _ = model.encode(pathways_b, drug_features_t[drug_b])
            loss = criterion(model.logits(latent), y_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            latent_val, _ = model.encode(x_val_t, drug_features_t[d_val_t])
            probability_val = torch.sigmoid(model.logits(latent_val)).cpu().numpy()

        try:
            auc = roc_auc_score(y_val, probability_val)
        except ValueError:
            auc = 0.0

        print(
            f"[source] epoch {epoch + 1:02d}/{cfg.source_epochs} "
            f"loss={running_loss / max(n_batches, 1):.4f} val_auc={auc:.4f}"
        )

        if auc > best_auc:
            best_auc = auc
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= cfg.source_patience:
                break

    if best_state is None:
        raise RuntimeError("Source pretraining did not produce a valid model state.")

    model.load_state_dict(best_state)
    return model, best_auc, idx_val


def adapt_to_target(
    model: ChemGuideModel,
    source_pathways: np.ndarray,
    source_drug_idx: np.ndarray,
    source_y: np.ndarray,
    source_sample_weights: np.ndarray,
    target_pathways: np.ndarray,
    target_drug_idx: int,
    drug_features_t: torch.Tensor,
    cfg: Config,
    device: torch.device,
):
    """Unsupervised/refinement stage for GSE112274 with pseudo-label class alignment."""
    target_x = torch.tensor(target_pathways, dtype=torch.float32, device=device)
    n_target = target_x.shape[0]
    target_feat = drug_features_t[target_drug_idx].unsqueeze(0)

    source_loader = DataLoader(
        TensorDataset(
            torch.tensor(source_pathways),
            torch.tensor(source_drug_idx),
            torch.tensor(source_y, dtype=torch.float32),
            torch.tensor(source_sample_weights, dtype=torch.float32),
        ),
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=seeded_generator(cfg.seed + 1),
    )

    pos_weight = torch.tensor(
        [(source_y == 0).sum() / max((source_y == 1).sum(), 1)],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    rng = np.random.default_rng(cfg.seed + 2)

    total_epochs = cfg.adapt_epochs + cfg.refine_epochs
    pseudo = None

    for epoch in range(total_epochs):
        in_refine = epoch >= cfg.adapt_epochs
        if in_refine and ((epoch - cfg.adapt_epochs) % cfg.pseudo_update_every == 0):
            pseudo = make_pseudo_labels(model, target_x, target_feat, cfg)

        model.train()
        aggregate = {"bce": 0.0, "align": 0.0, "manifold": 0.0, "total": 0.0}
        n_batches = 0

        for pathways_b, drug_b, y_b, weight_b in source_loader:
            pathways_b = pathways_b.to(device)
            drug_b = drug_b.to(device)
            y_b = y_b.to(device)
            weight_b = weight_b.to(device)

            source_z, _ = model.encode(pathways_b, drug_features_t[drug_b])
            loss_bce = criterion(model.logits(source_z), y_b)

            target_batch_size = min(cfg.target_batch_size, n_target)
            target_indices = torch.tensor(
                rng.choice(
                    n_target,
                    size=target_batch_size,
                    replace=(target_batch_size > n_target),
                ),
                dtype=torch.long,
                device=device,
            )
            target_b = target_x[target_indices]
            target_z, _ = model.encode(
                target_b, target_feat.expand(target_batch_size, -1)
            )

            loss_align = torch.tensor(0.0, device=device)
            if in_refine and pseudo is not None:
                class_loss = class_conditional_alignment(
                    model,
                    source_z,
                    y_b,
                    target_x,
                    target_feat,
                    target_indices,
                    pseudo,
                    source_weights=weight_b,
                )
                if class_loss is not None:
                    loss_align = cfg.lambda_align * class_loss

            loss_manifold = cfg.lambda_manifold * (
                manifold_loss(pathways_b, source_z, cfg.manifold_k)
                + manifold_loss(target_b, target_z, cfg.manifold_k)
            )

            loss = loss_bce + loss_align + loss_manifold
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            aggregate["bce"] += loss_bce.item()
            aggregate["align"] += float(loss_align)
            aggregate["manifold"] += float(loss_manifold)
            aggregate["total"] += loss.item()
            n_batches += 1

        phase = "refine" if in_refine else "adapt"
        pseudo_note = (
            ""
            if (not in_refine or pseudo is not None)
            else " | pseudo labels unavailable"
        )
        n_batches = max(n_batches, 1)
        print(
            f"[{phase}] epoch {epoch + 1:02d}/{total_epochs} "
            f"total={aggregate['total'] / n_batches:.4f} "
            f"bce={aggregate['bce'] / n_batches:.4f} "
            f"align={aggregate['align'] / n_batches:.4f} "
            f"manifold={aggregate['manifold'] / n_batches:.4f}{pseudo_note}"
        )

    return model
