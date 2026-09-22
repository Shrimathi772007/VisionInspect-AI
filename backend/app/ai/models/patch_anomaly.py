"""Patch-level anomaly detection on frozen CNN features (PatchCore-style nearest-neighbour and a
global-Gaussian / Mahalanobis variant).

    image -> frozen ResNet-18 -> mid-level feature maps (layer2 [+ layer3]) -> 3x3 local average pooling
    -> one feature vector per spatial patch -> distance to "normal" patches learned ONLY from train/good
    -> per-patch anomaly map -> aggregation (max, or mean of the top 1% of patches) -> scalar image score

Why this can succeed where whole-image pixel reconstruction fails: the score is (a) computed on
texture-aware features rather than raw pixels and (b) *local* - one abnormal patch is not averaged away
by hundreds of normal ones. Fitting is pure statistics on good training images: nothing is trained by
gradient descent and the backbone is never fine-tuned.

Deterministic on CPU: the only randomness is the coreset's seeded random projection / start index.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from app.ai.models.resnet18 import ResNet18, imagenet_normalize, load_pretrained_resnet18

AGG_MAX = "max"
AGG_TOP1PCT = "top1pct_mean"
AGG_TOP5PCT = "top5pct_mean"
AGG_TOP1PCT_MEDIAN = "top1pct_median"
AGG_TOP5PCT_MEDIAN = "top5pct_median"
AGGREGATIONS = (AGG_MAX, AGG_TOP1PCT, AGG_TOP5PCT, AGG_TOP1PCT_MEDIAN, AGG_TOP5PCT_MEDIAN)

SCORER_KNN = "knn"
SCORER_GAUSSIAN = "gaussian"


@torch.no_grad()
def extract_patch_features(
    extractor: ResNet18, images: torch.Tensor, layers: tuple[int, ...], batch_size: int = 8
) -> torch.Tensor:
    """[0,1] RGB NCHW images -> (N, h, w, C) patch features (concatenated, locally pooled layers)."""
    outputs = []
    for start in range(0, images.shape[0], batch_size):
        maps = extractor(imagenet_normalize(images[start : start + batch_size]), layers)
        pooled = [F.avg_pool2d(maps[layer], kernel_size=3, stride=1, padding=1) for layer in layers]
        size = pooled[0].shape[-2:]
        aligned = [p if p.shape[-2:] == size else F.interpolate(p, size=size, mode="bilinear", align_corners=False)
                   for p in pooled]
        outputs.append(torch.cat(aligned, dim=1).permute(0, 2, 3, 1).contiguous())
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def greedy_coreset_indices(features: torch.Tensor, n_select: int, seed: int, projection_dim: int = 64) -> torch.Tensor:
    """Deterministic greedy k-center subsampling (PatchCore's coreset) on a seeded random projection."""
    generator = torch.Generator().manual_seed(seed)
    n = features.shape[0]
    n_select = min(n_select, n)
    if features.shape[1] > projection_dim:
        projection = torch.randn(features.shape[1], projection_dim, generator=generator) / math.sqrt(projection_dim)
        z = features @ projection
    else:
        z = features
    z_sq = (z * z).sum(1)
    start = int(torch.randint(n, (1,), generator=generator))
    selected = [start]
    min_dist = (z_sq - 2 * (z @ z[start]) + z_sq[start]).clamp_(min=0)
    for _ in range(n_select - 1):
        index = int(torch.argmax(min_dist))
        selected.append(index)
        min_dist = torch.minimum(min_dist, (z_sq - 2 * (z @ z[index]) + z_sq[index]).clamp_(min=0))
    return torch.tensor(selected, dtype=torch.long)


class KNNPatchScorer:
    """Anomaly score of a patch = Euclidean distance to its nearest patch in the good-only memory bank."""

    kind = SCORER_KNN

    def __init__(self, bank: torch.Tensor):
        self.bank = bank.contiguous()

    @classmethod
    def fit(cls, train_patches: torch.Tensor, coreset_fraction: float, seed: int) -> "KNNPatchScorer":
        flat = train_patches.reshape(-1, train_patches.shape[-1])
        if coreset_fraction >= 1.0:
            return cls(flat.clone())
        n_select = max(1, int(round(flat.shape[0] * coreset_fraction)))
        return cls(flat[greedy_coreset_indices(flat, n_select, seed)].clone())

    @torch.no_grad()
    def patch_scores(self, patches: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
        n, h, w, c = patches.shape
        query = patches.reshape(-1, c)
        out = torch.empty(query.shape[0])
        for start in range(0, query.shape[0], chunk):
            out[start : start + chunk] = torch.cdist(query[start : start + chunk], self.bank).min(dim=1).values
        return out.reshape(n, h, w)

    def state(self) -> dict:
        return {"bank": self.bank}

    @classmethod
    def from_state(cls, state: dict) -> "KNNPatchScorer":
        return cls(state["bank"])

    @property
    def size_bytes(self) -> int:
        return self.bank.numel() * self.bank.element_size()


class GaussianPatchScorer:
    """Anomaly score of a patch = Mahalanobis distance to one Gaussian fitted on all good training patches."""

    kind = SCORER_GAUSSIAN

    def __init__(self, mean: torch.Tensor, precision: torch.Tensor):
        self.mean = mean
        self.precision = precision

    @classmethod
    def fit(cls, train_patches: torch.Tensor, shrinkage: float = 1e-2) -> "GaussianPatchScorer":
        x = train_patches.reshape(-1, train_patches.shape[-1]).double()
        mean = x.mean(0)
        centered = x - mean
        cov = centered.T @ centered / (x.shape[0] - 1)
        cov = cov + torch.eye(cov.shape[0], dtype=cov.dtype) * shrinkage * torch.diagonal(cov).mean()
        return cls(mean.float(), torch.linalg.inv(cov).float())

    @classmethod
    def from_statistics(cls, count: int, total: torch.Tensor, outer_total: torch.Tensor,
                        shrinkage: float = 1e-2) -> "GaussianPatchScorer":
        """Fit from precomputed sums over the training patches (count, sum x, sum x x^T; float64).

        Mathematically identical to `fit` on the same patches; lets cross-validation fits reuse per-image
        statistics instead of re-reading every feature map.
        """
        mean = total / count
        cov = (outer_total - count * torch.outer(mean, mean)) / (count - 1)
        cov = cov + torch.eye(cov.shape[0], dtype=cov.dtype) * shrinkage * torch.diagonal(cov).mean()
        return cls(mean.float(), torch.linalg.inv(cov).float())

    @torch.no_grad()
    def patch_scores(self, patches: torch.Tensor) -> torch.Tensor:
        n, h, w, c = patches.shape
        d = patches.reshape(-1, c) - self.mean
        return ((d @ self.precision) * d).sum(1).clamp_(min=0).sqrt().reshape(n, h, w)

    def state(self) -> dict:
        return {"mean": self.mean, "precision": self.precision}

    @classmethod
    def from_state(cls, state: dict) -> "GaussianPatchScorer":
        return cls(state["mean"], state["precision"])

    @property
    def size_bytes(self) -> int:
        return (self.mean.numel() + self.precision.numel()) * 4


def aggregate_scores(patch_scores: torch.Tensor, method: str) -> torch.Tensor:
    """(N, h, w) patch anomaly maps -> (N,) scalar image scores."""
    flat = patch_scores.flatten(1)
    if method == AGG_MAX:
        return flat.max(dim=1).values
    fractions = {AGG_TOP1PCT: 0.01, AGG_TOP5PCT: 0.05, AGG_TOP1PCT_MEDIAN: 0.01, AGG_TOP5PCT_MEDIAN: 0.05}
    if method in fractions:
        k = max(1, math.ceil(fractions[method] * flat.shape[1]))
        top = flat.topk(k, dim=1).values
        if method in (AGG_TOP1PCT, AGG_TOP5PCT):
            return top.mean(dim=1)
        return torch.quantile(top, 0.5, dim=1)  # interpolated median of the top-k patch scores
    raise ValueError(f"Unknown aggregation '{method}'.")


@dataclass
class PatchAnomalyDetector:
    """A complete, frozen image-level anomaly scorer: backbone + layers + input size + scorer + aggregation."""

    extractor: ResNet18
    layers: tuple[int, ...]
    image_size: int
    scorer: KNNPatchScorer | GaussianPatchScorer
    aggregation: str

    @torch.no_grad()
    def score_images(self, images: torch.Tensor) -> list[float]:
        """[0,1] RGB NCHW images at `image_size` -> anomaly score per image (higher = more anomalous)."""
        patches = extract_patch_features(self.extractor, images, self.layers)
        return [float(s) for s in aggregate_scores(self.scorer.patch_scores(patches), self.aggregation)]

    def config(self) -> dict:
        return {
            "detector": "patch_anomaly",
            "backbone": "resnet18_imagenet_frozen",
            "layers": list(self.layers),
            "image_size": self.image_size,
            "scorer": self.scorer.kind,
            "aggregation": self.aggregation,
            "feature_dim": int(self.scorer.bank.shape[1] if self.scorer.kind == SCORER_KNN else self.scorer.mean.shape[0]),
        }

    def save(self, directory: Path) -> Path:
        """Writes model_state.pt + model_config.json. The backbone weights are NOT duplicated: they are the
        separately hash-verified ImageNet file referenced by the config."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.scorer.state(), directory / "model_state.pt")
        (directory / "model_config.json").write_text(json.dumps(self.config(), indent=2), encoding="utf-8")
        return directory / "model_state.pt"

    @classmethod
    def load(cls, directory: Path, extractor: ResNet18 | None = None) -> "PatchAnomalyDetector":
        directory = Path(directory)
        config = json.loads((directory / "model_config.json").read_text(encoding="utf-8"))
        state = torch.load(directory / "model_state.pt", map_location="cpu", weights_only=True)
        scorer = (KNNPatchScorer if config["scorer"] == SCORER_KNN else GaussianPatchScorer).from_state(state)
        return cls(
            extractor=extractor if extractor is not None else load_pretrained_resnet18(),
            layers=tuple(config["layers"]),
            image_size=config["image_size"],
            scorer=scorer,
            aggregation=config["aggregation"],
        )
