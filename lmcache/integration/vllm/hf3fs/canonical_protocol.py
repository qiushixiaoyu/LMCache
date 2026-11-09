# SPDX-License-Identifier: Apache-2.0
"""Canonical byte-object validation shared by the bridge and MP extension."""

# Future
from __future__ import annotations

# Standard
from typing import Any
import hashlib
import re

MAX_BODY = 64 * 1024**2
MAX_OBJECTS = 256
KEY = re.compile(
    r"dsv41_canonical_v1@[0-9a-f]{64}@layer_(2|8|14|20)/(main_kv|indexer_k)@[0-9a-f]{32,128}\Z"
)


def validate(operation: str, rows: list[list[Any]]) -> None:
    """Reject malformed canonical identities, payload sizes and oversized batches."""
    if (
        operation not in ("exists", "get", "put")
        or not isinstance(rows, list)
        or not 0 < len(rows) <= MAX_OBJECTS
    ):
        raise ValueError("Invalid operation or batch size")
    total = 0
    for row in rows:
        if not isinstance(row, list) or len(row) != (3 if operation == "put" else 2):
            raise ValueError("Invalid object descriptor")
        key, size = row[:2]
        if not isinstance(key, str) or not KEY.fullmatch(key):
            raise ValueError("Invalid canonical key")
        if type(size) is not int or not 0 < size <= 2 * 1024**2:
            raise ValueError("Invalid byte size")
        if operation == "put" and (
            not isinstance(row[2], bytes) or len(row[2]) != size
        ):
            raise ValueError("Invalid payload length")
        total += size
    if total > MAX_BODY:
        raise ValueError("Batch exceeds byte budget")


def identity(canonical: str, size: int, chunk: int) -> Any:
    """Preserve the original bridge's model namespace and synthetic token IDs."""
    name = canonical.rsplit("@", 1)[0] + f"@bytes={size}"
    model = "dsv41-lmcache-" + hashlib.sha256(name.encode()).hexdigest()[:40]
    digest = hashlib.sha256(canonical.encode()).digest()
    return model, (list(digest) * ((chunk + 31) // 32))[:chunk]
