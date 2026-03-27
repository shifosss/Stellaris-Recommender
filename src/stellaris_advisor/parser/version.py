"""Stellaris version detection and validation."""

from __future__ import annotations

from typing import Any

SUPPORTED_VERSIONS: tuple[str, ...] = ("4.3",)


class UnsupportedVersionError(Exception):
    """Raised when the detected Stellaris version is not supported."""

    def __init__(self, detected: str, supported: tuple[str, ...]) -> None:
        self.detected = detected
        self.supported = supported
        super().__init__(
            f"Stellaris version {detected} is not supported. "
            f"Supported: {', '.join(supported)}"
        )


def detect_version(
    meta: dict[str, Any],
    supported: tuple[str, ...] = SUPPORTED_VERSIONS,
) -> str:
    """Extract major.minor version from meta dict.

    '4.3.0.0' → '4.3'. Raises UnsupportedVersionError if not in supported,
    or KeyError if 'version' key is missing.
    """
    raw = meta["version"]
    parts = str(raw).split(".")
    major_minor = f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else str(raw)

    if major_minor not in supported:
        raise UnsupportedVersionError(major_minor, supported)

    return major_minor
