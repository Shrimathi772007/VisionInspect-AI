"""Anomaly-detection model components (feature extractors and patch-level detectors).

Separate from app.ai.training.model (the pixel-reconstruction ConvAutoencoder, kept unchanged).
Everything here is inference-time PyTorch/NumPy only - no torchvision or other new dependency.
"""
