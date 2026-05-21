"""
soundaqnet
==========
Inferring affective quality from soundscape clips.

Adapted from SoundSCaper (https://github.com/Yuanbo2020/SoundSCaper).
Credits: Hou et al. (2026), IEEE Transactions on Multimedia.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("soundaqnet")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

# Run dependency version checks on every import.
# Emits warnings (not exceptions) so downstream code can still run if the
# user knowingly has a slightly different version installed.
from soundaqnet._compat import run_all_checks as _run_all_checks
_run_all_checks()

# Re-export the top-level public API so users can write:
#   from soundaqnet import SoundAQnet
#
# Use a lazy import so that `import soundaqnet` succeeds even when torch is
# not yet installed (e.g. during pip dependency resolution).  The heavy import
# only happens when the user actually accesses the SoundAQnet class.
def __getattr__(name: str):
    if name == "SoundAQnet":
        from soundaqnet.inference import SoundAQnet  # noqa: F401
        return SoundAQnet
    raise AttributeError(f"module 'soundaqnet' has no attribute {name!r}")

__all__ = ["SoundAQnet", "__version__"]
