"""PatchCore anomaly detector on a frozen WideResNet-50-2 (memory-bounded, deterministic).

    image -> crop224 / full256 preprocessing -> frozen WRN-50-2 -> layer2 + layer3
    -> 3x3 local average pooling, layer3 bilinearly upsampled to layer2's grid, concatenated (1536-d)
    -> k=1 nearest-neighbour distance to a seeded greedy-coreset memory bank of train/good patches
    -> per-patch anomaly scores -> image score (max, or mean of the top ceil(1%) patches)
                                -> anomaly map (bilinear upsample + Gaussian blur, sigma 4)

A separate class from app.ai.models.patch_anomaly.PatchAnomalyDetector, which serves the existing
ResNet-18 models and stays unchanged. Its greedy coreset and score aggregation are reused read-only, so
the coreset selection and the top-1% rule are identical in both detectors.

Memory: features are always written into pre-allocated tensors (never stacked from per-image lists), and
nearest-neighbour queries run in chunks of at most 256 patch rows.
"""

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from app.ai.models.patch_anomaly import AGG_MAX, AGG_TOP1PCT, aggregate_scores, greedy_coreset_indices
from app.ai.models.resnet18 import file_sha256
from app.ai.models.wide_resnet50 import WRN50_2_SHA256, load_pretrained_wide_resnet50_2
from app.ai.preprocessing.patchcore_preprocess import MODE_CROP224, MODES, input_size

BACKBONE_WRN50_2 = "wide_resnet50_2_imagenet_frozen"
AGGREGATIONS = (AGG_MAX, AGG_TOP1PCT)  # "max", "top1pct_mean"
FEATURE_STRIDE = 8  # layer2 is 1/8 of the input resolution
MAX_QUERY_CHUNK = 256
BLUR_SIGMA = 4.0
BLUR_TAPS = 33  # 2 * ceil(4 * sigma) + 1
STATE_FILENAME = "model_state.pt"
CONFIG_FILENAME = "model_config.json"
FORMAT_VERSION = 1


class PatchCoreModelError(Exception):
    """A saved PatchCore model is incomplete, altered, or was built for different backbone weights."""


@dataclass(frozen=True)
class PatchCoreConfig:
    backbone: str = BACKBONE_WRN50_2
    weights_sha256: str = WRN50_2_SHA256
    layers: tuple[int, ...] = (2, 3)
    neighbourhood: str = "avg_pool_3x3"
    preprocessing: str = MODE_CROP224
    input_size: int = field(default=-1)
    coreset_ratio: float = 0.10
    coreset_seed: int = 0
    projection_dim: int = 64
    k: int = 1
    aggregation: str = AGG_MAX

    def __post_init__(self):
        if self.preprocessing not in MODES:
            raise ValueError(f"Unknown preprocessing '{self.preprocessing}'. Expected one of {MODES}.")
        expected_size = input_size(self.preprocessing)
        if self.input_size == -1:
            object.__setattr__(self, "input_size", expected_size)
        elif self.input_size != expected_size:
            raise ValueError(f"input_size {self.input_size} does not match preprocessing '{self.preprocessing}'.")
        if self.aggregation not in AGGREGATIONS:
            raise ValueError(f"Unknown aggregation '{self.aggregation}'. Expected one of {AGGREGATIONS}.")
        if self.k != 1:
            raise ValueError("Only k=1 nearest-neighbour scoring is supported.")
        if not 0.0 < self.coreset_ratio <= 1.0:
            raise ValueError("coreset_ratio must be in (0, 1].")
        object.__setattr__(self, "layers", tuple(self.layers))

    @property
    def grid_side(self) -> int:
        return self.input_size // FEATURE_STRIDE

    @property
    def patches_per_image(self) -> int:
        return self.grid_side**2

    def to_dict(self) -> dict:
        data = asdict(self)
        data["layers"] = list(self.layers)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "PatchCoreConfig":
        return cls(**{name: data[name] for name in cls.__dataclass_fields__})


def gaussian_kernel_1d(sigma: float = BLUR_SIGMA, taps: int = BLUR_TAPS) -> torch.Tensor:
    t = torch.arange(taps, dtype=torch.float32) - (taps - 1) / 2
    kernel = torch.exp(-(t**2) / (2 * sigma**2))
    return kernel / kernel.sum()


class PatchCoreDetector:
    """Frozen backbone + config + memory bank. `extractor` must map an ImageNet-normalized NCHW batch to
    {layer: feature map} like WideResNet50_2.forward; by default the verified pretrained WRN-50-2 is loaded."""

    def __init__(self, config: PatchCoreConfig | None = None, extractor: nn.Module | None = None,
                 bank: torch.Tensor | None = None):
        self.config = config or PatchCoreConfig()
        self.extractor = extractor if extractor is not None else load_pretrained_wide_resnet50_2()
        self.bank = bank.contiguous() if bank is not None else None

    # ------------------------------------------------------------------ features
    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """(B, 3, S, S) normalized images -> (B, P, D) patch features (P = (S/8)^2; D = 1536 for WRN-50-2)."""
        maps = self.extractor(images, self.config.layers)
        pooled = [F.avg_pool2d(maps[layer], kernel_size=3, stride=1, padding=1) for layer in self.config.layers]
        size = pooled[0].shape[-2:]
        aligned = [p if p.shape[-2:] == size else F.interpolate(p, size=size, mode="bilinear", align_corners=False)
                   for p in pooled]
        features = torch.cat(aligned, dim=1)
        batch, dim = features.shape[:2]
        return features.permute(0, 2, 3, 1).reshape(batch, -1, dim).contiguous()

    @torch.no_grad()
    def extract_features_into(self, images: Iterable[torch.Tensor], count: int, batch_size: int = 8) -> torch.Tensor:
        """Features for `count` preprocessed (3, S, S) images, written chunk by chunk into ONE pre-allocated
        (count, P, D) tensor. Neither the inputs nor the features are ever stacked from per-image lists."""
        side = self.config.input_size
        batch = torch.empty((batch_size, 3, side, side), dtype=torch.float32)
        out = None
        written = 0
        filled = 0

        def flush(n: int):
            nonlocal out, written
            features = self.extract_features(batch[:n])
            if out is None:
                out = torch.empty((count, features.shape[1], features.shape[2]), dtype=torch.float32)
            out[written : written + n] = features
            written += n

        for image in images:
            if written + filled >= count:
                raise ValueError(f"More than {count} images were supplied.")
            batch[filled] = image
            filled += 1
            if filled == batch_size:
                flush(filled)
                filled = 0
        if filled:
            flush(filled)
        if written != count:
            raise ValueError(f"Expected {count} images, got {written}.")
        return out

    # ------------------------------------------------------------------ memory bank
    @torch.no_grad()
    def fit_from_features(self, features: torch.Tensor) -> torch.Tensor:
        """Seeded greedy-coreset memory bank from (N, P, D) train/good features (same selection rule and size
        rounding as the existing detector). Deterministic for identical features and seed."""
        flat = features.reshape(-1, features.shape[-1])
        if self.config.coreset_ratio >= 1.0:
            self.bank = flat.clone()
        else:
            n_select = max(1, int(round(flat.shape[0] * self.config.coreset_ratio)))
            indices = greedy_coreset_indices(flat, n_select, self.config.coreset_seed, self.config.projection_dim)
            self.bank = flat[indices].clone()
        return self.bank

    # ------------------------------------------------------------------ scoring
    @torch.no_grad()
    def patch_scores_from_features(self, features: torch.Tensor, chunk: int = MAX_QUERY_CHUNK) -> torch.Tensor:
        """(B, P, D) -> (B, P) Euclidean distance of every patch to its nearest bank patch (k=1), computed in
        query chunks of at most 256 rows so memory stays bounded."""
        if self.bank is None:
            raise PatchCoreModelError("The detector has no memory bank; fit or load it first.")
        if not 1 <= chunk <= MAX_QUERY_CHUNK:
            raise ValueError(f"chunk must be between 1 and {MAX_QUERY_CHUNK}.")
        batch, patches, dim = features.shape
        query = features.reshape(-1, dim)
        out = torch.empty(query.shape[0], dtype=torch.float32)
        for start in range(0, query.shape[0], chunk):
            out[start : start + chunk] = torch.cdist(query[start : start + chunk], self.bank).min(dim=1).values
        return out.reshape(batch, patches)

    def image_scores(self, patch_scores: torch.Tensor, aggregation: str | None = None) -> torch.Tensor:
        """(B, P) -> (B,): "max", or "top1pct_mean" = mean of the top ceil(0.01 * P) patch scores (8 of 784,
        11 of 1024) - the existing detector's aggregation, reused unchanged."""
        aggregation = aggregation or self.config.aggregation
        if aggregation not in AGGREGATIONS:
            raise ValueError(f"Unknown aggregation '{aggregation}'. Expected one of {AGGREGATIONS}.")
        return aggregate_scores(patch_scores, aggregation)

    @torch.no_grad()
    def anomaly_map(self, patch_scores: torch.Tensor, out_size: int | None = None) -> torch.Tensor:
        """(B, P) -> (B, H, W) float32: patch grid -> bilinear upsample to out_size -> Gaussian blur (sigma 4,
        33-tap separable kernel, reflect padding). Both steps are convex combinations, so the map never exceeds
        the largest patch score."""
        out_size = out_size or self.config.input_size
        batch, patches = patch_scores.shape
        side = math.isqrt(patches)
        if side * side != patches:
            raise ValueError(f"{patches} patch scores do not form a square grid.")
        grid = patch_scores.reshape(batch, 1, side, side).float()
        up = F.interpolate(grid, size=(out_size, out_size), mode="bilinear", align_corners=False)
        kernel = gaussian_kernel_1d()
        radius = (BLUR_TAPS - 1) // 2
        blurred = F.pad(up, (radius, radius, radius, radius), mode="reflect")
        blurred = F.conv2d(blurred, kernel.view(1, 1, 1, -1))
        blurred = F.conv2d(blurred, kernel.view(1, 1, -1, 1))
        return blurred[:, 0].float()

    @torch.no_grad()
    def score_images(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, 3, S, S) normalized images -> (image scores (B,), anomaly maps (B, S, S)) in one pass."""
        patch_scores = self.patch_scores_from_features(self.extract_features(images))
        return self.image_scores(patch_scores), self.anomaly_map(patch_scores)

    # ------------------------------------------------------------------ persistence
    def save(self, directory: Path) -> Path:
        """Writes model_state.pt (the bank) and model_config.json (config + bank shape + SHA-256 of the state
        file). The backbone weights are not duplicated: the config records their pinned SHA-256."""
        if self.bank is None:
            raise PatchCoreModelError("Nothing to save: the detector has no memory bank.")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        state_path = directory / STATE_FILENAME
        torch.save({"bank": self.bank}, state_path)
        record = {
            "detector": "patchcore",
            "format_version": FORMAT_VERSION,
            **self.config.to_dict(),
            "bank_shape": list(self.bank.shape),
            "state_sha256": file_sha256(state_path),
        }
        (directory / CONFIG_FILENAME).write_text(json.dumps(record, indent=2), encoding="utf-8")
        return state_path

    @classmethod
    def load(cls, directory: Path, extractor: nn.Module | None = None) -> "PatchCoreDetector":
        """Loads a saved detector, refusing it if its recorded backbone weights hash is not the pinned one or if
        the state file does not match its recorded SHA-256. The state is read with weights_only=True."""
        directory = Path(directory)
        state_path, config_path = directory / STATE_FILENAME, directory / CONFIG_FILENAME
        if not state_path.is_file() or not config_path.is_file():
            raise PatchCoreModelError(f"Incomplete PatchCore model in {directory.name}: missing state or config.")
        record = json.loads(config_path.read_text(encoding="utf-8"))
        if record.get("detector") != "patchcore":
            raise PatchCoreModelError("Not a PatchCore model configuration.")
        if record.get("weights_sha256") != WRN50_2_SHA256:
            raise PatchCoreModelError(
                f"Model was built for backbone weights {record.get('weights_sha256')!r}, not the pinned {WRN50_2_SHA256}."
            )
        if file_sha256(state_path) != record.get("state_sha256"):
            raise PatchCoreModelError("Model state file does not match its recorded SHA-256.")
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        bank = state["bank"]
        if list(bank.shape) != record.get("bank_shape"):
            raise PatchCoreModelError("Model state bank shape does not match the recorded shape.")
        return cls(PatchCoreConfig.from_dict(record), extractor=extractor, bank=bank)
