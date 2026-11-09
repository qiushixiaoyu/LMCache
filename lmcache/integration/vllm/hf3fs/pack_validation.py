# SPDX-License-Identifier: Apache-2.0
"""Pack header bounds are mandatory; hash checks are optional diagnostics."""

# Future
from __future__ import annotations

# Standard
from typing import Any
import hashlib

# Local
from .pack_format import BODY_SIZE, HEADER, LAYOUT_HASH, MAGIC, PACK_SIZE


def valid_pack(group: str, data: Any, verify_hashes: bool = False) -> bool:
    """Check structural bounds and optionally the layout, identity and body hashes."""
    if len(data) != PACK_SIZE:
        return False
    magic, layout, identity, digest, size = HEADER.unpack_from(data)
    if magic != MAGIC or size != BODY_SIZE:
        return False
    if not verify_hashes:
        return True
    return (
        layout == LAYOUT_HASH
        and identity == hashlib.sha256(group.encode()).digest()
        and digest == hashlib.sha256(data[HEADER.size :]).digest()
    )
