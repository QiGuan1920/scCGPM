"""Domain-alignment and geometry-preservation objectives."""

import torch


def coral_loss(
    source_z: torch.Tensor,
    target_z: torch.Tensor,
    source_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weighted first-order mean + second-order covariance alignment."""
    dim = source_z.shape[1]

    def weighted_mean(z: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
        if weights is None:
            return z.mean(0)
        weights = weights / (weights.sum() + 1e-12)
        return (weights.view(-1, 1) * z).sum(0)

    def weighted_cov(z: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
        n = z.shape[0]
        if weights is None:
            centered = z - z.mean(0, keepdim=True)
            return (centered.t() @ centered) / max(n - 1, 1)

        weights = weights / (weights.sum() + 1e-12)
        mean = (weights.view(-1, 1) * z).sum(0, keepdim=True)
        centered = z - mean
        return (centered.t() * weights.view(1, -1)) @ centered

    mean_loss = (
        (weighted_mean(source_z, source_weights) - weighted_mean(target_z, None)) ** 2
    ).sum() / dim
    covariance_loss = (
        (weighted_cov(source_z, source_weights) - weighted_cov(target_z, None)) ** 2
    ).sum() / dim

    return mean_loss + covariance_loss


def manifold_loss(
    original: torch.Tensor,
    latent: torch.Tensor,
    k: int = 10,
) -> torch.Tensor:
    """Preserve local kNN geometry from pathway space in the latent space."""
    batch_size = original.shape[0]
    if batch_size < k + 1:
        return latent.sum() * 0.0

    with torch.no_grad():
        d2_original = torch.cdist(original, original) ** 2
        positive = d2_original[d2_original > 0]
        sigma = (
            positive.median() + 1e-8
            if positive.numel()
            else torch.tensor(1.0, device=original.device)
        )
        weights = torch.exp(-d2_original / sigma)
        weights.fill_diagonal_(0.0)

        neighbor_idx = torch.topk(weights, k, dim=1).indices
        mask = torch.zeros_like(weights)
        mask.scatter_(1, neighbor_idx, 1.0)
        weights = weights * mask
        weights = torch.maximum(weights, weights.t())

    d2_latent = torch.cdist(latent, latent) ** 2
    return (weights * d2_latent).sum() / (weights.sum() + 1e-8)
