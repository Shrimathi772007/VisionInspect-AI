"""Model artifact (checkpoint) storage.

Artifacts are saved outside the `app/` source tree, in a sibling directory
next to backend/storage/, backend/venv/, etc. - never inside Git-tracked
source. See the root .gitignore for the corresponding ignore rule.
"""

from pathlib import Path

import torch
from torch import nn

from app.ai.training.model import DEFAULT_LATENT_CHANNELS, ConvAutoencoder, build_model

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent.parent  # .../backend
ARTIFACTS_ROOT = (BACKEND_DIR / "ai_models").resolve()


def get_model_path(category: str, model_name: str = "autoencoder") -> Path:
    """Where a trained model for `category` is/should be stored."""
    return ARTIFACTS_ROOT / category / f"{model_name}.pt"


def save_model(model: nn.Module, path: Path) -> Path:
    """Save a model's weights (state_dict) to `path`, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path


def load_model(path: Path, model: nn.Module) -> nn.Module:
    """Load saved weights into `model` (an already-constructed instance of the right architecture).

    Returns the same model, in eval() mode, for convenience.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {path}")

    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_model_for_category(
    category: str, model_name: str = "autoencoder", latent_channels: int = DEFAULT_LATENT_CHANNELS
) -> ConvAutoencoder:
    """Convenience: build a fresh ConvAutoencoder and load the saved weights for `category`.

    This is the function later inference code should use to get a ready-to-use model.
    """
    path = get_model_path(category, model_name)
    model = build_model(latent_channels=latent_channels)
    return load_model(path, model)
