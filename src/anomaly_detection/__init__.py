"""Small, tested interfaces for the repository's handwritten detectors."""

from .mahalanobis import MahalanobisDetector
from .pca import PCAReconstructionDetector

__all__ = ["MahalanobisDetector", "PCAReconstructionDetector"]
__version__ = "0.1.0a0"
