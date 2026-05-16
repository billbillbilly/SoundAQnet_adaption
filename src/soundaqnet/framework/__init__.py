"""
soundaqnet.framework
====================
Neural network architecture and training utilities for SoundAQnet.
"""

from soundaqnet.framework import config  # noqa: F401
from soundaqnet.framework.models_pytorch import SoundAQnet  # noqa: F401

__all__ = ["config", "SoundAQnet"]
