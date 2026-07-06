"""
soundaqnet._compat
==================
Runtime version-compatibility checks.

This module is imported by ``soundaqnet/__init__.py`` on every package import.
It emits ``warnings.warn`` for soft mismatches and raises ``ImportError`` for
hard incompatibilities (e.g. numpy 2.x breaking audio libs).

Validated dependency matrix
---------------------------
| Package       | Required version | Notes                              |
|---------------|------------------|------------------------------------|
| torch         | >=2.1.0          | GPU builds need custom index URL   |
| torchaudio    | >=2.1.0          | Must match torch major.minor       |
| torchvision   | >=0.16.0         | Must match torch major.minor       |
| numpy         | <2.0.0           | v2 breaks librosa/soundfile/h5py   |
"""

from __future__ import annotations

import sys
import warnings
from typing import Optional

# ── helpers ──────────────────────────────────────────────────────────────────


def _parse_version(v: str) -> tuple[int, ...]:
    """Convert '2.1.0' → (2, 1, 0).  Ignores pre-release suffixes."""
    return tuple(int(x) for x in v.split("+")[0].split(".")[:3] if x.isdigit())


def _check_exact(pkg: str, required: str, installed: str) -> None:
    if _parse_version(installed) != _parse_version(required):
        warnings.warn(
            f"soundaqnet was validated with {pkg}=={required} but "
            f"{installed} is installed.  Unexpected behaviour may occur.\n"
            f"  Fix: pip install {pkg}=={required}",
            stacklevel=4,
        )


def _check_max(pkg: str, max_excl: str, installed: str) -> None:
    if _parse_version(installed) >= _parse_version(max_excl):
        raise ImportError(
            f"soundaqnet requires {pkg}<{max_excl} but {installed} is installed.\n"
            f"  Fix: pip install '{pkg}<{max_excl}'"
        )


# ── torch / torchaudio / torchvision ─────────────────────────────────────────


def check_torch() -> Optional[str]:
    """Return the installed torch version string, or None if not found."""
    try:
        import torch

        ver = torch.__version__
    except ImportError:
        warnings.warn(
            "torch is not installed.  soundaqnet requires torch>=2.1.0.\n"
            "  Windows / Linux (CUDA 12.1):\n"
            "    pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
            "  macOS / CPU-only:\n"
            "    pip install torch",
            stacklevel=3,
        )
        return None
    if _parse_version(ver) < _parse_version("2.1.0"):
        warnings.warn(
            f"soundaqnet requires torch>=2.1.0 but {ver} is installed.\n"
            f"  Fix: pip install --upgrade torch",
            stacklevel=4,
        )
    return ver


def check_torchaudio() -> None:
    try:
        import torchaudio

        ver = torchaudio.__version__
        if _parse_version(ver) < _parse_version("2.1.0"):
            warnings.warn(
                f"soundaqnet requires torchaudio>=2.1.0 but {ver} is installed.\n"
                f"  Fix: pip install --upgrade torchaudio",
                stacklevel=3,
            )
    except ImportError:
        warnings.warn(
            "torchaudio is not installed.  soundaqnet requires torchaudio>=2.1.0.\n"
            "  Install it with the same --index-url you used for torch.",
            stacklevel=3,
        )


def check_torchvision() -> None:
    try:
        import torchvision

        ver = torchvision.__version__
        if _parse_version(ver) < _parse_version("0.16.0"):
            warnings.warn(
                f"soundaqnet requires torchvision>=0.16.0 but {ver} is installed.\n"
                f"  Fix: pip install --upgrade torchvision",
                stacklevel=3,
            )
    except ImportError:
        warnings.warn(
            "torchvision is not installed.  soundaqnet requires torchvision>=0.16.0.\n"
            "  Install it with the same --index-url you used for torch.",
            stacklevel=3,
        )


# ── numpy ─────────────────────────────────────────────────────────────────────


def check_numpy() -> None:
    try:
        import numpy as np

        _check_max("numpy", "2.0.0", np.__version__)
    except ImportError:
        pass  # numpy is pulled in by other deps; ImportError is unexpected


# ── platform (ISO 532-1 loudness) ─────────────────────────────────────────────


def check_loudness_platform() -> None:
    """Warn if running on macOS/Linux without mosqito installed."""
    if sys.platform != "win32":
        try:
            import mosqito  # noqa: F401
        except ImportError:
            warnings.warn(
                "mosqito is not installed.  On macOS/Linux, mosqito is required "
                "for ISO 532-1 loudness extraction (replacing the Windows-only "
                "ISO_532-1.exe binary).\n"
                "  Fix: pip install 'mosqito>=1.2.0'",
                stacklevel=3,
            )


# ── main entry point ──────────────────────────────────────────────────────────


def run_all_checks() -> None:
    """Run all compatibility checks.  Called once at package import time."""
    check_numpy()
    check_torch()
    check_torchaudio()
    check_torchvision()
    check_loudness_platform()
