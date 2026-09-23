"""ChemGuide neural architecture."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config


class ChemGuideModel(nn.Module):
    """Drug-guided pathway encoder with response-prior-aware dynamic gating."""

    def __init__(self, n_pathways: int, drug_feat_dim: int, cfg: Config):
        super().__init__()
        self.n_pathways = n_pathways

        self.drug_encoder = nn.Sequential(
            nn.Linear(drug_feat_dim, cfg.drug_out),
            nn.ReLU(),
        )
        self.gate = nn.Linear(cfg.drug_out + 1, n_pathways)
        self.encoder = nn.Sequential(
            nn.Linear(n_pathways + cfg.drug_out, cfg.adapter_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.adapter_hidden, cfg.z_dim),
        )
        self.classifier = nn.Linear(cfg.z_dim, 1)

        self.register_buffer("proto_pos", torch.zeros(n_pathways))
        self.register_buffer("proto_neg", torch.zeros(n_pathways))

    def set_prototypes(self, proto_pos: np.ndarray, proto_neg: np.ndarray) -> None:
        self.proto_pos.copy_(torch.as_tensor(proto_pos, dtype=torch.float32))
        self.proto_neg.copy_(torch.as_tensor(proto_neg, dtype=torch.float32))

    def encode(self, pathway_scores: torch.Tensor, drug_features: torch.Tensor):
        drug_embedding = self.drug_encoder(drug_features)

        sim_pos = F.cosine_similarity(
            pathway_scores, self.proto_pos[None], dim=1, eps=1e-8
        )
        sim_neg = F.cosine_similarity(
            pathway_scores, self.proto_neg[None], dim=1, eps=1e-8
        )
        response_prior = (sim_pos - sim_neg).unsqueeze(1)

        pathway_gate = torch.sigmoid(
            self.gate(torch.cat([drug_embedding, response_prior], dim=1))
        )
        gated_pathways = pathway_gate * pathway_scores
        latent = self.encoder(torch.cat([gated_pathways, drug_embedding], dim=1))
        return latent, pathway_gate

    def logits(self, latent: torch.Tensor) -> torch.Tensor:
        return self.classifier(latent).squeeze(-1)
